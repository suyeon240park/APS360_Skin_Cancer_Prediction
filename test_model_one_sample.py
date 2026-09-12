import argparse
from pathlib import Path

import torch
from PIL import Image

from model import CLASS_NAMES, get_transforms, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the APS360 skin-lesion classifier on one image."
    )
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to a trained model state_dict (default: best_model.pth).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu, cuda, or mps.",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return get_transforms()(rgb_image).unsqueeze(0)


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    model = load_model(args.model, device=args.device)
    image_tensor = load_image(args.image).to(args.device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_index].item())

    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
