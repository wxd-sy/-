import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from config import Config
from target_cnn_spec import get_total_param_count
from data_loader import get_dataloaders


class BaselineCNN(nn.Module):
    def __init__(self, cnn_spec_name: str = "default"):
        super().__init__()
        self.cnn_spec_name = cnn_spec_name
        if cnn_spec_name == "default":
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
        elif cnn_spec_name == "large":
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        if self.cnn_spec_name == "large":
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_baseline(cfg: Config):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = get_dataloaders(
        cfg.dataset, cfg.batch_size, cfg.num_workers
    )

    model = BaselineCNN(cfg.cnn_spec).to(device)
    baseline_params = sum(p.numel() for p in model.parameters())
    print(f"\nBaseline CNN params: {baseline_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Baseline Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0*correct/total:.2f}%"})
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)
        val_acc = 100.0 * val_correct / val_total
        best_acc = max(best_acc, val_acc)
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    print(f"\nBaseline best accuracy: {best_acc:.2f}%")
    return model, baseline_params, best_acc


def compare(cfg: Config):
    print("\n" + "=" * 60)
    print("Baseline vs Mapping Network Comparison")
    print("=" * 60)

    from trainer import train as train_mapping

    print("\n>>> Training Baseline CNN...")
    baseline_model, baseline_params, baseline_acc = train_baseline(cfg)

    target_params_baseline = get_total_param_count(cfg.cnn_spec)
    print(f"Baseline CNN params (spec): {target_params_baseline:,}")

    print("\n>>> Training Mapping Network...")
    mapping_model, mapping_acc = train_mapping(cfg)

    mapping_trainable = mapping_model.count_trainable_params()
    mapping_target = mapping_model.count_target_params()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Baseline':>15} {'Mapping':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Test Accuracy (%)':<30} {baseline_acc:>15.2f} {mapping_acc:>15.2f}")
    print(f"  {'Trainable Params':<30} {baseline_params:>15,} {mapping_trainable:>15,}")
    print(f"  {'Target Network Params':<30} {target_params_baseline:>15,} {mapping_target:>15,}")
    print(f"  {'Compression Ratio':<30} {'1x':>15} {mapping_target/(mapping_trainable or 1):>15.1f}x")
    print("=" * 60)

    return {
        "baseline_acc": baseline_acc,
        "mapping_acc": mapping_acc,
        "baseline_params": baseline_params,
        "mapping_trainable": mapping_trainable,
        "mapping_target": mapping_target,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--cnn_spec", type=str, default="large")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config(
        dataset=args.dataset,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        cnn_spec=args.cnn_spec,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        device=args.device,
    )
    results = compare(cfg)
