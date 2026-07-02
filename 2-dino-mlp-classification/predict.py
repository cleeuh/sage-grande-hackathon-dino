import sys
from pathlib import Path

import torch
from PIL import Image

from model import DinoClassifier

TEST_DIR = Path("./test")
HEAD = "head.pt"


def choose_image() -> Path:
    images = sorted(
        p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No .jpg/.png images found in {TEST_DIR}/")

    print("Select an image to classify:")
    for i, path in enumerate(images, 1):
        print(f"  {i}. {path.name}")

    while True:
        choice = input(f"Enter a number (1-{len(images)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(images):
            return images[int(choice) - 1]
        print("Invalid selection, try again.")


def main() -> None:
    # Get image path from args or cli menu
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.is_file():
            raise SystemExit(f"No such file: {image_path}")
    else:
        image_path = choose_image()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a ready-to-predict model (backbone + trained head + transform).
    model = DinoClassifier.from_checkpoint(HEAD, device=device)

    label, prob, probs = model.predict(Image.open(image_path))

    print(f"\n{image_path.name}")
    print(f"Prediction: {label}  (p={prob:.3f})")
    for cls, p in zip(model.classes, probs.tolist()):
        print(f"  {cls:20s} {p:.3f}")


if __name__ == "__main__":
    main()
