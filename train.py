import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from models import CDSFNet
from utils import (
    ForgettingAwareClassBalancer,
    ImageSamplesDataset,
    accuracy_at_k,
    build_sample_weights,
    build_train_samples,
    evaluate_model,
    load_or_create_split_three_way,
    print_results,
)


REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pseudo_lambda_at_epoch(epoch_idx: int, warmup_epochs: int, max_lambda: float) -> float:
    if warmup_epochs <= 0:
        return max_lambda
    ratio = min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
    return max_lambda * ratio


def build_transforms(img_size: int = 640):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 12),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ft_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, train_ft_transform, eval_transform


def train_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    grad_accum: int = 2,
    label_smoothing: float = 0.0,
    pseudo_lambda: float = 0.0,
    amp_enabled: bool = True,
    fcb_weighting=None,
):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    top1_correct = 0.0
    top3_correct = 0.0
    total = 0

    optimizer.zero_grad()
    for step, (images, labels, is_pseudo, sample_ids) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        is_pseudo = is_pseudo.to(device).bool()

        weight_each = torch.ones(labels.shape[0], device=device, dtype=torch.float32)
        if fcb_weighting is not None:
            weight_each = weight_each * fcb_weighting.compute_weights(sample_ids, labels)
        if torch.any(is_pseudo):
            weight_each = torch.where(is_pseudo, weight_each * pseudo_lambda, weight_each)

        def weighted_ce(logits):
            ce_each = F.cross_entropy(
                logits,
                labels,
                reduction="none",
                label_smoothing=label_smoothing,
            )
            return (ce_each * weight_each).mean()

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss_ce = weighted_ce(logits)
            total_loss = loss_ce / grad_accum

        scaler.scale(total_loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += total_loss.item() * grad_accum
        running_ce += loss_ce.item()

        preds = logits.detach().argmax(dim=1)
        if fcb_weighting is not None:
            fcb_weighting.update(sample_ids, labels.detach(), preds)

        c1, c3 = accuracy_at_k(logits, labels)
        top1_correct += c1
        top3_correct += c3
        total += labels.size(0)

    n_batches = max(1, len(dataloader))
    return {
        "loss": running_loss / n_batches,
        "ce": running_ce / n_batches,
        "acc1": top1_correct / max(1, total),
        "acc3": top3_correct / max(1, total),
    }


def validate(model, dataloader, device, label_smoothing: float = 0.0):
    model.eval()
    running_loss = 0.0
    top1_correct = 0.0
    top3_correct = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(
                logits,
                labels,
                reduction="mean",
                label_smoothing=label_smoothing,
            )
            running_loss += loss.item()
            c1, c3 = accuracy_at_k(logits, labels)
            top1_correct += c1
            top3_correct += c3
            total += labels.size(0)

    n_batches = max(1, len(dataloader))
    return running_loss / n_batches, top1_correct / max(1, total), top3_correct / max(1, total)


def save_ckpt(path: str, model) -> None:
    torch.save({"model": model.state_dict()}, path)


def load_ckpt(path: str, model, device) -> None:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
    else:
        model.load_state_dict(obj)


def make_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="CDSF-Net training")
    parser.add_argument("--data_dir", type=str, default="images")
    parser.add_argument("--pseudo_root", type=str, default="images_pseudo")
    parser.add_argument("--split_file", type=str, default="splits/split_seed42_75_10_15_no_xixia.json")
    parser.add_argument("--output_dir", type=str, default="runs/default")
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--use_pseudo", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ft_epochs", type=int, default=100)
    parser.add_argument("--s1_patience", type=int, default=25)
    parser.add_argument("--s2_patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ft_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=2e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--es_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--patch_dim", type=int, default=256)
    parser.add_argument("--fuse_dim", type=int, default=512)
    parser.add_argument("--patch_topk", type=int, default=16)
    parser.add_argument("--dasf_heads", type=int, default=4)
    parser.add_argument("--dasf_window_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--pseudo_weight", type=float, default=0.4)
    parser.add_argument("--pseudo_lambda_max", type=float, default=0.4)
    parser.add_argument("--pseudo_lambda_warmup", type=int, default=20)

    parser.add_argument("--use_fcb", type=int, default=1)
    parser.add_argument("--fcb_class_scale", type=float, default=1.5)
    parser.add_argument("--fcb_sample_scale", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    pseudo_root = resolve_path(args.pseudo_root)
    split_file = resolve_path(args.split_file)
    output_dir = resolve_path(args.output_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 72)
    print("CDSF-Net: ConvNeXt + CSFR + DASF + FCB")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Data Dir: {data_dir}")
    print(f"Pseudo Root: {pseudo_root}")
    print(f"Split File: {split_file}")
    print(f"Output Dir: {output_dir}")
    print(f"Use Pseudo: {args.use_pseudo}")
    print(f"Epochs: {args.epochs} + {args.ft_epochs}")
    print("=" * 72 + "\n")

    train_samples, val_samples, test_samples, class_names = load_or_create_split_three_way(
        data_dir=data_dir,
        split_path=split_file,
        project_root=str(REPO_ROOT),
        random_state=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    all_train_samples, clean_train_samples_flagged, pseudo_samples = build_train_samples(
        clean_train_samples=train_samples,
        class_names=class_names,
        pseudo_root=pseudo_root,
        use_pseudo=args.use_pseudo,
    )

    train_transform, train_ft_transform, eval_transform = build_transforms(args.img_size)

    train_dataset = ImageSamplesDataset(all_train_samples, transform=train_transform, return_flag=True)
    val_dataset = ImageSamplesDataset(val_samples, transform=eval_transform, return_flag=False)
    test_dataset = ImageSamplesDataset(test_samples, transform=eval_transform, return_flag=False)

    weights_np, _ = build_sample_weights(
        all_train_samples,
        class_names,
        pseudo_weight=args.pseudo_weight,
        small_boost=2.0,
    )
    sampler = WeightedRandomSampler(weights_np, num_samples=len(weights_np), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CDSFNet(
        num_classes=len(class_names),
        patch_dim=args.patch_dim,
        fuse_dim=args.fuse_dim,
        patch_topk=args.patch_topk,
        dasf_heads=args.dasf_heads,
        dasf_window_size=args.dasf_window_size,
        dropout=args.dropout,
    ).to(device)

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

    s1_ckpt = os.path.join(output_dir, "checkpoints", "cdsf_net_s1.pth")
    final_ckpt = os.path.join(output_dir, "checkpoints", "cdsf_net_final.pth")

    optimizer = make_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    save_ckpt(s1_ckpt, model)
    best_val_loss = float("inf")
    s1_patience_counter = 0

    fcb_weighting = None
    if args.use_fcb:
        fcb_weighting = ForgettingAwareClassBalancer(
            num_classes=len(class_names),
            class_scale=args.fcb_class_scale,
            sample_scale=args.fcb_sample_scale,
            max_sample_boost=3.0,
        )

    for epoch in range(args.epochs):
        lam = pseudo_lambda_at_epoch(epoch, args.pseudo_lambda_warmup, args.pseudo_lambda_max) if args.use_pseudo else 0.0
        tr = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum=args.grad_accum,
            label_smoothing=args.label_smoothing,
            pseudo_lambda=lam,
            amp_enabled=(device.type == "cuda"),
            fcb_weighting=fcb_weighting,
        )
        val_loss, val_acc1, val_acc3 = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            label_smoothing=args.label_smoothing,
        )
        scheduler.step()

        print(
            f"S1 Ep {epoch + 1:3d} | TrainLoss {tr['loss']:.4f} (CE {tr['ce']:.4f}) "
            f"| ValLoss {val_loss:.4f} | PseudoLam {lam:.3f} "
            f"| Val {val_acc1:.2%} (T3 {val_acc3:.2%})",
            flush=True,
        )

        if val_loss < (best_val_loss - args.es_min_delta):
            best_val_loss = val_loss
            s1_patience_counter = 0
            save_ckpt(s1_ckpt, model)
        else:
            s1_patience_counter += 1
            if s1_patience_counter >= args.s1_patience:
                print(f"Stage 1 early stopping at epoch {epoch + 1}", flush=True)
                break

    load_ckpt(s1_ckpt, model, device=device)

    clean_only_train = [(path, label, 0) for path, label, _ in clean_train_samples_flagged]
    train_ft_dataset = ImageSamplesDataset(clean_only_train, transform=train_ft_transform, return_flag=True)
    clean_weights_np, _ = build_sample_weights(
        clean_only_train,
        class_names,
        pseudo_weight=1.0,
        small_boost=2.0,
    )
    sampler_ft = WeightedRandomSampler(clean_weights_np, num_samples=len(clean_weights_np), replacement=True)
    train_ft_loader = DataLoader(
        train_ft_dataset,
        batch_size=args.batch_size,
        sampler=sampler_ft,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = make_optimizer(model=model, lr=args.ft_lr, weight_decay=1e-2)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    baseline_val_loss, baseline_val_acc1, baseline_val_acc3 = validate(
        model=model,
        dataloader=val_loader,
        device=device,
        label_smoothing=args.label_smoothing,
    )
    best_ft_val_loss = baseline_val_loss
    s2_patience_counter = 0
    save_ckpt(final_ckpt, model)
    print(
        f"Stage2 baseline | ValLoss {baseline_val_loss:.4f} | "
        f"Val {baseline_val_acc1:.2%} (T3 {baseline_val_acc3:.2%})",
        flush=True,
    )

    fcb_weighting_ft = None
    if args.use_fcb:
        fcb_weighting_ft = ForgettingAwareClassBalancer(
            num_classes=len(class_names),
            class_scale=args.fcb_class_scale,
            sample_scale=args.fcb_sample_scale,
            max_sample_boost=3.0,
        )

    for epoch in range(args.ft_epochs):
        tr = train_epoch(
            model=model,
            dataloader=train_ft_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum=args.grad_accum,
            label_smoothing=args.label_smoothing,
            pseudo_lambda=0.0,
            amp_enabled=(device.type == "cuda"),
            fcb_weighting=fcb_weighting_ft,
        )
        val_loss, val_acc1, val_acc3 = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            label_smoothing=args.label_smoothing,
        )

        print(
            f"S2 Ep {epoch + 1:3d} | TrainLoss {tr['loss']:.4f} (CE {tr['ce']:.4f}) "
            f"| ValLoss {val_loss:.4f} | Val {val_acc1:.2%} (T3 {val_acc3:.2%})",
            flush=True,
        )

        if val_loss < (best_ft_val_loss - args.es_min_delta):
            best_ft_val_loss = val_loss
            s2_patience_counter = 0
            save_ckpt(final_ckpt, model)
        else:
            s2_patience_counter += 1
            if s2_patience_counter >= args.s2_patience:
                print(f"Stage 2 early stopping at epoch {epoch + 1}", flush=True)
                break

    load_ckpt(final_ckpt, model, device=device)
    results = evaluate_model(model, test_loader, device, class_names)
    print_results(results, class_names)

    results_dict = {
        "model": "CDSF-Net",
        "description": "ConvNeXt + CSFR + DASF + FCB",
        "top1_acc": float(results["top1_acc"]),
        "top3_acc": float(results["top3_acc"]),
        "f1_macro": float(results["f1_macro"]),
        "f1_weighted": float(results["f1_weighted"]),
        "per_class_acc": {key: float(value) for key, value in results["per_class_acc"].items()},
        "num_clean_train": len(clean_train_samples_flagged),
        "num_pseudo_train": len(pseudo_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "class_names": class_names,
        "hyperparameters": vars(args),
    }
    with open(os.path.join(output_dir, "results", "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, ensure_ascii=False, indent=2)
    np.save(os.path.join(output_dir, "results", "confusion_matrix.npy"), results["confusion_matrix"])


if __name__ == "__main__":
    main()
