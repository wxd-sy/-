import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict

from rtdetr.mapping_backbone import MappingBackbone, _sanitize
from rtdetr.spec import get_stage_groups


def _delta_magnitude_per_stage(mapping: MappingBackbone) -> dict:
    mags = {}
    stages = get_stage_groups(mapping.depth)
    for stage in stages:
        total_mag = 0.0
        count = 0
        for name in mapping._name_list:
            if mapping.conv_stage.get(name) == stage:
                z = mapping._get_latent(stage)
                proj = mapping.projections[_sanitize(name)]
                with torch.no_grad():
                    raw = proj(z, mapping.alpha, return_smoothness=False)
                total_mag += raw.abs().mean().item()
                count += 1
        mags[stage] = total_mag / max(count, 1)
    return mags


def _qdelta_magnitude(mapping: MappingBackbone) -> Dict[str, float]:
    if not mapping.use_query_mapping:
        return {}
    with torch.no_grad():
        qp_delta = mapping.generate_querypos_delta(return_smoothness=False)
    return {
        name: d["weight"].abs().mean().item()
        for name, d in qp_delta.items()
    }


def run_backbone_diagnostic(mapping: MappingBackbone, device: torch.device):
    mapping.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640, device=device)
        out = mapping(dummy, return_smoothness=False)
        if isinstance(out, tuple):
            feats = out[0]
        else:
            feats = out
        print("  Backbone feature shapes:")
        for i, f in enumerate(feats):
            print(f"    feat[{i}]: {tuple(f.shape)}  mean={f.mean().item():.4f}  std={f.std().item():.4f}")
        n_train = mapping.count_trainable_params()
        n_target = mapping.count_target_params()
        print(f"  Trainable params: {n_train:,}")
        print(f"  Target params (conv + query_pos_head): {n_target:,}")
        if n_train > 0:
            print(f"  Compression ratio: {n_target / n_train:.1f}x")


def train_rtdetr_backbone(
    mapping: MappingBackbone,
    encoder: Optional[nn.Module],
    decoder: Optional[nn.Module],
    criterion: Optional[nn.Module],
    train_loader,
    cfg,
    device: torch.device,
    eval_fn=None,
    eval_every: int = 0,
    is_baseline: bool = False,
    qp_orig_state: Optional[Dict[str, torch.Tensor]] = None,
):
    use_qm = getattr(cfg, "use_query_mapping", False)
    lambda_align = getattr(cfg, "lambda_align", 0.01)

    if is_baseline:
        print("\n  *** BASELINE MODE: all params frozen, loss=baseline ***")
        mapping.eval()
        total_base = 0.0
        n = 0
        with torch.no_grad():
            for images, targets in tqdm(train_loader, desc="Baseline eval"):
                images = images.to(device)
                td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                feats = mapping(images, return_smoothness=False)
                if isinstance(feats, tuple):
                    feats = feats[0] if len(feats) == 2 else feats
                encoded = encoder(feats)
                out = decoder(encoded, td)
                loss_dict = criterion(out, td)
                total_base += sum(v.item() for v in loss_dict.values() if isinstance(v, torch.Tensor))
                n += 1
        avg_base = total_base / max(1, n)
        print(f"  Baseline avg task_loss: {avg_base:.4f}")
        mags = _delta_magnitude_per_stage(mapping)
        print(f"  ΔW magnitude per stage: {mags}")
        return mapping

    optimizer = optim.AdamW(mapping.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs, pct_start=0.05,
    )

    run_backbone_diagnostic(mapping, device)
    mags0 = _delta_magnitude_per_stage(mapping)
    print(f"  Initial backbone ΔW magnitude per stage: {mags0}")
    if use_qm:
        qmags0 = _qdelta_magnitude(mapping)
        print(f"  Initial query_pos ΔW magnitude: {qmags0}")

    if use_qm and lambda_align > 0:
        print("\n  Pre-computing reference features for alignment loss...")
        mapping.eval()
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device)
                feats = mapping(images, return_smoothness=False)
                if isinstance(feats, tuple):
                    feats = feats[0]
                mapping.set_ref_features(feats)
                break
        print("  Reference features stored.")

    best_loss = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        mapping.train()
        total_loss = 0.0
        total_task = 0.0
        total_align = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, targets in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            qp_delta = None
            qp_wrapper = None
            orig_qph = None

            if use_qm:
                feats, s_val, qp_delta = mapping(images, return_smoothness=True, return_qdelta=True)
            else:
                feats, s_val = mapping(images, return_smoothness=True)

            pbar_desc = {}

            if encoder is not None and decoder is not None and criterion is not None:
                td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                if use_qm and qp_delta is not None:
                    from rtdetr.encoder_decoder import make_querypos_wrapper
                    orig_qph = decoder.query_pos_head
                    qp_wrapper = make_querypos_wrapper(decoder, qp_delta)
                    decoder.query_pos_head = qp_wrapper

                encoded = encoder(feats)
                decoder_out = decoder(encoded, td)
                loss_dict = criterion(decoder_out, td)
                task_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

                if use_qm and orig_qph is not None:
                    decoder.query_pos_head = orig_qph
            else:
                task_loss = sum(f.abs().mean() for f in feats) * 0.0001

            align_loss = mapping.compute_align_loss(feats) if use_qm else torch.tensor(0.0, device=device)

            total = task_loss + cfg.lambda_smoothness * s_val
            if use_qm and lambda_align > 0:
                total = total + lambda_align * align_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(mapping.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += total.item()
            if isinstance(task_loss, torch.Tensor):
                total_task += task_loss.item()
            total_align += align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss
            n_batches += 1

            if n_batches % cfg.log_interval == 0:
                pbar_desc["loss"] = f"{total.item():.4f}"
                pbar_desc["task"] = f"{task_loss.item():.4f}" if isinstance(task_loss, torch.Tensor) else f"{task_loss:.4f}"
                pbar_desc["smooth"] = f"{s_val.item():.4f}"
                if use_qm:
                    pbar_desc["align"] = f"{align_loss.item():.4f}" if isinstance(align_loss, torch.Tensor) else f"{align_loss:.4f}"
                pbar.set_postfix(pbar_desc)

        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        lr_val = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  task={avg_task:.4f}  "
              f"align={total_align/n_batches:.4f}  lr={lr_val:.2e}")

        if epoch % max(1, eval_every) == 0 or epoch == cfg.epochs:
            mags = _delta_magnitude_per_stage(mapping)
            mag_str = "  ".join(f"{k}={v:.4f}" for k, v in mags.items())
            print(f"  backbone ΔW mag: {mag_str}")
            if use_qm:
                qmags = _qdelta_magnitude(mapping)
                qmag_str = "  ".join(f"{k}={v:.6f}" for k, v in qmags.items())
                print(f"  query_pos ΔW mag: {qmag_str}")
            if eval_fn:
                metrics = eval_fn(mapping, encoder, decoder, train_loader, device)
                print(f"  mAP: {metrics}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch, "model": mapping.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(cfg.save_dir, f"{cfg.exp_name}_best.pt"))

    return mapping


def train_full_finetune(
    mapping, encoder, decoder, criterion, train_loader,
    cfg, device, eval_fn=None, eval_every=0,
):
    print("\n  *** FULL FINE-TUNE MODE: unfreezing all parameters ***")

    backbone = mapping
    for p in backbone.parameters():
        p.requires_grad = True

    params = list(backbone.parameters())
    has_detector = encoder is not None and decoder is not None
    if has_detector:
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True
        params += list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    os.makedirs(cfg.save_dir, exist_ok=True)

    n_params = sum(p.numel() for p in params)
    print(f"  Trainable params: {n_params:,}")

    for epoch in range(1, cfg.epochs + 1):
        backbone.train()
        if has_detector:
            encoder.train()
            decoder.train()

        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"FullFT Epoch {epoch}")

        for images, targets in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            feats = backbone(images, return_smoothness=False)
            if isinstance(feats, tuple):
                feats = feats[0]

            if has_detector:
                td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]
                encoded = encoder(feats)
                decoder_out = decoder(encoded, td)
                loss_dict = criterion(decoder_out, td)
                task_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            else:
                task_loss = sum(f.abs().mean() for f in feats) * 0.0001

            task_loss.backward()
            optimizer.step()
            total_loss += task_loss.item()
            n_batches += 1

            if n_batches % cfg.log_interval == 0:
                pbar.set_postfix({"loss": f"{task_loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / n_batches
        lr_val = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  lr={lr_val:.2e}")

        if eval_fn and (epoch % max(1, eval_every) == 0 or epoch == cfg.epochs):
            metrics = eval_fn(backbone, encoder, decoder, train_loader, device)
            print(f"  mAP: {metrics}")

        ckpt = {"epoch": epoch, "backbone": backbone.state_dict(), "optimizer": optimizer.state_dict()}
        if has_detector:
            ckpt["encoder"] = encoder.state_dict()
            ckpt["decoder"] = decoder.state_dict()
        torch.save(ckpt, os.path.join(cfg.save_dir, f"{cfg.exp_name}_best.pt"))


def train_lora(
    mapping, encoder, decoder, criterion, train_loader,
    cfg, device, eval_fn=None, eval_every=0,
):
    print("\n  *** LoRA MODE ***")
    has_detector = encoder is not None and decoder is not None
    if not has_detector:
        print("  No detector loaded, running backbone-only dummy training")
        return

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("  peft not installed. Install with: pip install peft")
        return

    # Freeze all first
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    for p in mapping.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=2 * cfg.lora_rank,
        target_modules=["in_proj_weight", "in_proj_bias",
                        "out_proj.weight", "out_proj.bias",
                        "sampling_offsets", "value_proj",
                        "attention_weights", "output_proj"],
        lora_dropout=0.0,
    )
    encoder = get_peft_model(encoder, lora_config)
    decoder = get_peft_model(decoder, lora_config)

    n_trainable = (sum(p.numel() for p in encoder.parameters() if p.requires_grad) +
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    print(f"  LoRA rank={cfg.lora_rank}, trainable params: {n_trainable:,}")

    params = (list(encoder.parameters()) +
              list(decoder.parameters()))
    optimizer = optim.AdamW([p for p in params if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    os.makedirs(cfg.save_dir, exist_ok=True)

    backbone = mapping

    for epoch in range(1, cfg.epochs + 1):
        backbone.eval()
        encoder.train()
        decoder.train()

        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"LoRA Epoch {epoch}")

        for images, targets in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]
            with torch.no_grad():
                feats = backbone(images, return_smoothness=False)
                if isinstance(feats, tuple):
                    feats = feats[0]

            encoded = encoder(feats)
            decoder_out = decoder(encoded, td)
            loss_dict = criterion(decoder_out, td)
            task_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

            task_loss.backward()
            optimizer.step()
            total_loss += task_loss.item()
            n_batches += 1

            if n_batches % cfg.log_interval == 0:
                pbar.set_postfix({"loss": f"{task_loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / n_batches
        lr_val = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  lr={lr_val:.2e}")

        if eval_fn and (epoch % max(1, eval_every) == 0 or epoch == cfg.epochs):
            metrics = eval_fn(backbone, encoder, decoder, train_loader, device)
            print(f"  mAP: {metrics}")

        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join(cfg.save_dir, f"{cfg.exp_name}_best.pt"))
