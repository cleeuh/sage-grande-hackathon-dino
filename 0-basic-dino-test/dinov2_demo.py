# Basic Demonstration of how to get the model up and running

import io
import urllib.request

import torch
from PIL import Image
from torchvision import transforms

MODEL_DIR       = "../model/dinov2"                                 # path
MODEL           = "dinov2_vitl14"                                   # version of dinov2
MODEL_WEIGHTS   = "../model/weights/dinov2_vitl14_pretrain.pth"     # dinov2 backbone weights file

IMAGE           = "../data/Dog-Park-Graphic.png"    # Test image for demo
SET_IMAGE_SIZE  = 518                                               # any multiple of the patch size (14) works.  DINOV2 primarily trained on 518 x 518


def load_image(path: str) -> Image.Image:
    """Opens an image from path"""
    
    return Image.open(path).convert("RGB")

def build_transform(image_size: int) -> transforms.Compose:
    """Image Preprocessing"""

    return transforms.Compose(
        [
            # Resize so image_size lines up with DINOv2's patch grid
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),

            # Crop the center so the final image is exactly image_size x image_size
            transforms.CenterCrop(image_size),

            # Convert PIL image, shape (Height, Width, Channel) with pixel intensitiy values [0, 255],
            # to a tensor of shape (Channel, Height, Width) scaled to pixel intensitiy values [0.0, 1.0]
            transforms.ToTensor(),

            # Normalize to ImageNet mean/std to prevent distribution shift from what DINOv2 was pretrained on
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def main() -> None:
    # Catch error
    if SET_IMAGE_SIZE % 14 != 0:
        raise SystemExit("IMAGE_SIZE must be a multiple of 14 (DINOv2 patch size)")

    # Debug (using GPU or cpu?)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load Model
    model = torch.hub.load(MODEL_DIR, MODEL, source="local", pretrained=False)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))

    # Move model to device and disable dropout/batchnorm training behavior
    model = model.to(device).eval()

    # Load and preprocess the input image
    image = load_image(IMAGE)
    transform = build_transform(SET_IMAGE_SIZE)

    # unsqueeze(0) adds a batch dimension; DINOv2 expects input shaped
    # (Batch, Channel, Height, Width), even for a single image.
    tensor = transform(image).unsqueeze(0).to(device)

    # "Just run the model forward" mode - skips the bookkeeping needed for
    # training/backprop, since we're only doing inference here, not training.
    with torch.inference_mode():
        # Global image embedding (CLS token).
        cls_embedding = model(tensor)  # (1, embed_dim)

        # Dense per-patch features.
        out = model.forward_features(tensor)
        patch_tokens = out["x_norm_patchtokens"][0]  # (num_patches, embed_dim)

    print(f"CLS embedding shape: {tuple(cls_embedding.shape)}") # (batch, embed_dim)
    print(f"Patch tokens shape:  {tuple(patch_tokens.shape)}") # (patches, embed_dim)


if __name__ == "__main__":
    main()
