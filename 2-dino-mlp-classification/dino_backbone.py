import torch
from torchvision import transforms

MODEL_DIR       = "../model/dinov2"
MODEL           = "dinov2_vitl14"
MODEL_WEIGHTS = {
    "dinov2_vitl14": "../model/weights/dinov2_vitl14_pretrain.pth",
}

# DINOv2 backbones -> embedding dim. All use patch size 14.
EMBED_DIMS = {
    # "dinov2_vits14": 384,
    # "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    # "dinov2_vitg14": 1536,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(name: str, device: str) -> torch.nn.Module:
    """Load DinoV2 Backbone"""

    # Look to 0-basic-dino-test for detailed explainations
    if name not in EMBED_DIMS:
        raise ValueError(f"unknown model {name!r}; choose from {list(EMBED_DIMS)}")
    if name not in MODEL_WEIGHTS:
        raise ValueError(f"no local weights for {name!r}; have {list(MODEL_WEIGHTS)}")

    model = torch.hub.load(MODEL_DIR, name, source="local", pretrained=False)
    model.load_state_dict(torch.load(MODEL_WEIGHTS[name], map_location=device, weights_only=True))
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False) # Freezes layers
        
    embed_dim = EMBED_DIMS[name] # Embedding dimentionality from dictionary

    return model, embed_dim


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
