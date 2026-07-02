# This is a demonstration of the "information" contained within the DinoV2 patch embeddings.

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

MODEL_DIR       = "../model/dinov2"                                 # path
MODEL           = "dinov2_vitl14"                                   # version of dinov2
MODEL_WEIGHTS   = "../model/weights/dinov2_vitl14_pretrain.pth"     # dinov2 backbone weights file

IMAGE           = "../data/Dog-Park-Graphic.png"    # Test image for demo
SET_IMAGE_SIZE  = 518                                               # any multiple of the patch size (14) works.  DINOV2 primarily trained on 518 x 518
OUT             = "dinov2_pca.png"                                  # image output


def load_image(path: str) -> Image.Image:
    """Opens an image from path"""

    return Image.open(path).convert("RGB")

def build_transform(image_size: int) -> transforms.Compose:
    """Image Preprocessing"""

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def pca_rgb(patch_tokens: torch.Tensor, grid: int) -> np.ndarray:
    """PCA reduces dimensionality by finding the most prominent directions. (in this case, from dinov2 patch outputs)
    to demonstrate shared semantic similarity.  Learn more: https://setosa.io/ev/principal-component-analysis/    
    """
    
    # (num_patches, embed_dim) on CPU as float32 for sklearn.
    x = patch_tokens.float().cpu().numpy()

    # Fit PCA and project each patch onto its top-3 principal components.
    # PCA centers the features (subtracts the mean) internally.
    proj = PCA(n_components=3, svd_solver="full").fit_transform(x)  # (num_patches, 3)

    # Normalize each channel to [0, 1] for display.
    proj = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-8)

    # Reshape the flat patch sequence back into the (grid, grid) spatial layout.
    img = proj.reshape(grid, grid, 3)
    return (img * 255).astype(np.uint8)


def main() -> None:
    if SET_IMAGE_SIZE % 14 != 0:
        raise SystemExit("IMAGE_SIZE must be a multiple of 14 (DINOv2 patch size)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Loading Model
    model = torch.hub.load(MODEL_DIR, MODEL, source="local", pretrained=False)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    model = model.to(device).eval()

    # Test Model
    image = load_image(IMAGE)
    transform = build_transform(SET_IMAGE_SIZE)
    tensor = transform(image).unsqueeze(0).to(device)

    # Get patch tokens
    with torch.inference_mode():
        out = model.forward_features(tensor)
        patch_tokens = out["x_norm_patchtokens"][0]  # (num_patches, embed_dim)


    ############ The New Stuff: Semantic Patch Similarity via PCA ##############

    # The patch tokens form a (grid x grid) square; verify the count matches.
    grid = SET_IMAGE_SIZE // 14

    # Save at native patch resolution (grid x grid) - one pixel per patch.
    pca_img = pca_rgb(patch_tokens, grid)
    Image.fromarray(pca_img).save(OUT)
    print(f"Saved PCA feature visualization to {OUT} ({grid}x{grid})")


if __name__ == "__main__":
    main()
