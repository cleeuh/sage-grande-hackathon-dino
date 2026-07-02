import torch
import torch.nn as nn
import torch.nn.functional as F

from dino_backbone import EMBED_DIMS, build_transform, load_backbone


class DinoClassifier(nn.Module):
    """Whole Model using frozen DinoV2 as feature extractor with trainable MLP head."""

    def __init__(self, model_name: str, num_classes: int,
                 hidden_dim: int = 256, device: str = "cpu") -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone, embed_dim = load_backbone(model_name, device)  # layers are frozen in here, and get back cls dimensionality (embed_dim)


        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),  # linear: w·x + b        (input to model dim size must match prior layer output dim size)
            nn.ReLU(),                         # relu:   activation()
                                               # ----------------------
                                               # activation( w·x + b ) == relu(linear)
            nn.Linear(hidden_dim, num_classes),
        )
        self.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network (using in both training and inference)"""
        with torch.no_grad():
            features = self.backbone(images)  # get CLS embedding (semantic meaning of the entire image)
        return self.head(features)

    def train(self, mode: bool = True) -> "DinoClassifier":
        """Set training mode or eval mode"""

        super().train(mode)  # sets everyhting in the model to trainning mode
        self.backbone.eval() # reset backbone to eval so its dropout / stochastic-depth layers stay off and it yields the same (deterministic) features every pass
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "DinoClassifier":
        """load model from a head checkpoint saved by train.py"""
        ckpt = torch.load(path, map_location=device)                                 # load our train weights + metadata
        model = cls(ckpt["model"], num_classes=len(ckpt["classes"]), device=device)  # inits the base model
        model.head.load_state_dict(ckpt["state_dict"])                               # apply our trained weights
        model.classes = ckpt["classes"]                                              # set the classes to match
        model.transform = build_transform(ckpt["image_size"])                        # set the image preprocessor
        model.eval()                                                                 # set as eval mode
        return model

    @torch.inference_mode()
    def predict(self, image):
        """Classify a single PIL image -> (label, probability, all_probs tensor).

        Softmax lives here (not in ``forward``) so training keeps seeing raw
        logits for CrossEntropyLoss, while callers get probabilities for free.
        """
        device = next(self.head.parameters()).device                            # current device that model lives on
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(device)   # apply preprocessing to image and move to device
        probs = F.softmax(self.forward(tensor), dim=1)[0]                       # apply softmax to make into probability
        idx = int(probs.argmax())                                               # index of highest probability
        return self.classes[idx], float(probs[idx]), probs                      # return predicted class label, confidence (probabilities), and probability vector
