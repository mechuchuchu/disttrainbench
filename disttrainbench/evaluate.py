"""Evaluate one final checkpoint on CIFAR-100 test after training has ended."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision import datasets, transforms

from .model import ResNet20


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must be a non-empty state_dict or contain 'state_dict'.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("Checkpoint state_dict must map string names to tensors.")
    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    return checkpoint


def evaluate(checkpoint_path: Path, data_root: Path, batch_size: int, workers: int) -> dict:
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    model = ResNet20(num_classes=100)
    model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = datasets.CIFAR100(
        root=str(data_root), train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    correct = 0
    count = 0
    loss_total = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss_total += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.numel()

    return {
        "status": "completed",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CIFAR-100 test split",
        "test_examples": count,
        "correct": correct,
        "top1_accuracy": correct / count,
        "top1_accuracy_percent": 100.0 * correct / count,
        "mean_cross_entropy": loss_total / count,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "architecture": "ResNet20-CIFAR100-3x3-projection-shortcuts-v1",
        "input_normalization": {"mean": MEAN, "std": STD},
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.loader_workers < 0:
        parser.error("batch size must be positive and loader workers cannot be negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint not found: {args.checkpoint}")

    result = evaluate(args.checkpoint, args.data_root, args.batch_size, args.loader_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
        temporary = Path(out.name)
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
