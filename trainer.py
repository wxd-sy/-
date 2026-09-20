import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from config import Config
from mapping_network import MappingNetwork
from functional_cnn import functional_cnn_forward
from losses import compute_loss


def train_epoch(mapping, optimizer, scheduler, train_loader, cfg, device):
    mapping.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Train")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        generated_weights, smooth_val = mapping(return_smoothness=True)
        logits = functional_cnn_forward(x, generated_weights, cfg.cnn_spec)

        loss_dict = compute_loss(
            logits, y, mapping, x, functional_cnn_forward,
            cfg.cnn_spec,
            lambda_stability=cfg.lambda_stability,
            stability_sigma=cfg.stability_sigma,
            lambda_smoothness=cfg.lambda_smoothness,
            lambda_alignment=cfg.lambda_alignment,
            smoothness_val=smooth_val,
        )

        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(mapping.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_dict["total"].item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total_samples += y.size(0)

        if batch_idx % cfg.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "task": f"{loss_dict['task'].item():.4f}",
                "stab": f"{loss_dict['stability'].item():.6f}",
                "smth": f"{loss_dict['smoothness'].item():.6f}",
                "algn": f"{loss_dict['alignment'].item():.4f}",
                "acc": f"{100.0 * correct / total_samples:.2f}%",
            })

    return total_loss / len(train_loader), 100.0 * correct / total_samples


@torch.no_grad()
def validate(mapping, test_loader, cfg, device):
    mapping.eval()
    correct = 0
    total_samples = 0
    total_loss = 0.0

    for x, y in tqdm(test_loader, desc="Val"):
        x, y = x.to(device), y.to(device)

        generated_weights = mapping()
        logits = functional_cnn_forward(x, generated_weights, cfg.cnn_spec)
        loss = torch.nn.functional.cross_entropy(logits, y)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total_samples += y.size(0)

    return total_loss / len(test_loader), 100.0 * correct / total_samples


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    from data_loader import get_dataloaders
    train_loader, test_loader = get_dataloaders(
        cfg.dataset, cfg.batch_size, cfg.num_workers
    )

    mapping = MappingNetwork(
        latent_dim=cfg.latent_dim,
        cnn_spec_name=cfg.cnn_spec,
        alpha=cfg.alpha,
        block_size=cfg.block_size,
        use_tanh=cfg.use_tanh,
        cache_projections=cfg.cache_projections,
        latent_init_std=cfg.latent_init_std,
        output_gain=cfg.output_gain,
    ).to(device)

    trainable = mapping.count_trainable_params()
    target = mapping.count_target_params()
    print(f"Latent dim: {cfg.latent_dim}")
    print(f"Trainable params: {trainable:,}")
    print(f"Target CNN params: {target:,}")
    print(f"Compression ratio: {target / trainable:.1f}x")

    optimizer = optim.AdamW(
        mapping.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs,
        pct_start=0.05,
    )

    print("\n--- Pre-training tanh saturation diagnostic ---")
    with torch.no_grad():
        for name, proj in mapping.projections.items():
            g = torch.Generator(device=device).manual_seed(proj._seed.item())
            Wo = torch.randn(proj.out_dim, mapping.latent_dim,
                             generator=g, device=device, dtype=torch.float32)
            Wo = Wo / (Wo.norm(p=2, dim=1, keepdim=True) + 1e-8)
            raw = F.linear(mapping.z, Wo) + mapping.alpha * mapping.z.square().mean()
            in_range = ((raw > -2) & (raw < 2)).float().mean().item()
            print(f"  [{name}] raw in (-2,2): {in_range*100:.1f}%  "
                  f"|z|_2^2={mapping.z.square().sum().item():.6f}")
        x_sample, _ = next(iter(train_loader))
        x_sample = x_sample[:4].to(device)
        w = mapping()
        logits_sample = functional_cnn_forward(x_sample, w, cfg.cnn_spec)
        print(f"  Initial logits std: {logits_sample.std().item():.4f}")
    print("---\n")

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n--- Epoch {epoch}/{cfg.epochs} (lr={scheduler.get_last_lr()[0]:.2e}) ---")
        train_loss, train_acc = train_epoch(mapping, optimizer, scheduler, train_loader, cfg, device)
        val_loss, val_acc = validate(mapping, test_loader, cfg, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": mapping.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "cfg": cfg,
            }, os.path.join(cfg.save_dir, f"{cfg.exp_name}_best.pt"))
            print(f"  [Best model saved] Acc: {best_acc:.2f}%")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")
    return mapping, best_acc
