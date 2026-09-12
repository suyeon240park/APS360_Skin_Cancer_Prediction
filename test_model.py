import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import get_transforms, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the APS360 skin-lesion classifier on an ImageFolder dataset."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing one subdirectory per class, as required by ImageFolder.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to a trained model state_dict (default: best_model.pth).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu, cuda, or mps.",
    )
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
) -> tuple[int, int]:
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return correct, total


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    dataset = datasets.ImageFolder(root=args.data_dir, transform=get_transforms())
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(
        args.model,
        num_classes=len(dataset.classes),
        device=args.device,
    )
    correct, total = evaluate(model, data_loader, args.device)

    print(f"Classes: {dataset.classes}")
    print(f"Samples: {total}")
    if total:
        print(f"Accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()
