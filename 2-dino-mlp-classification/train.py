import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dino_backbone import build_transform
from model import DinoClassifier
from prepare_data import build_splits

# source class folders (RealWaste). Labels are assigned by sorted basename.
DATASET_SRC = "../data/waste/realwaste-main/RealWaste"
CLASSES = ["Paper", "Metal", "Glass"]   # train on n classes from DATASET_SRC
VAL_FRAC = 0.1                          # dataset split
TEST_FRAC = 0.1                         # dataset split
SEED = 0                                # randomization seed

MODEL = "dinov2_vitl14"                 # only backbone with local weights (see dino_backbone.MODEL_WEIGHTS)
IMAGE_SIZE = 224                        # keep as a multiple of 14
EPOCHS = 3
BATCH_SIZE = 32                         # for minibatching if device has vram limitations, also improves per epoch training
LR = 1e-3                               # learning rate
OUT = "head.pt"                         # save learned weights
NUM_WORKERS = 4                         # set to 0 to debug worker crashes (surfaces the real exception)

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Data Loader class to load in the data"""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)


def build_transform_train(image_size: int) -> transforms.Compose:
    """Train-time preprocessing/augmentation"""
    return transforms.Compose(
        [
            # crops and flips to encourage model invariance and better generalization
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


@torch.inference_mode()
def evaluate(model: DinoClassifier, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def train_model(model: DinoClassifier, train_loader, val_loader, classes, device) -> float:
    """Train the head for EPOCHS, saving the best-by-val-accuracy checkpoint to
    OUT. Returns the best validation accuracy."""
    # Only the head is trainable; the backbone is frozen inside DinoClassifier.
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()               # clear last batch's gradients (they accumulate otherwise)

            # Forward Pass
            logits = model(images)              # raw per-class scores (softmax to get probabilities)
            loss = criterion(logits, labels)    # the loss: one scalar to represnt how wrong this forward pass batch was

            # Backprop
            loss.backward()                     # backprop: from the one loss scalar, work out each weight/bias's gradient (how much that parameter contributed to the error)
            optimizer.step()                    # nudge each weight opposite its gradient
            running += loss.item() * labels.size(0)  # sum loss over samples (xN) for the epoch average

        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.head.state_dict(),
                        "classes": classes,
                        "model": MODEL,
                        "image_size": IMAGE_SIZE}, OUT)

    return best_acc


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, classes = build_splits(
        [f"{DATASET_SRC}/{c}" for c in CLASSES],
        train_transform=build_transform_train(IMAGE_SIZE),
        eval_transform=build_transform(IMAGE_SIZE),
        val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED,
    )
    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_ds, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_ds, BATCH_SIZE, shuffle=False)
    print(f"Classes: {classes}")
    print(f"Split sizes -> train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    model = DinoClassifier(MODEL, num_classes=len(classes), device=device)
    best_acc = train_model(model, train_loader, val_loader, classes, device)
    print(f"Best val accuracy: {best_acc:.4f}. Head saved to {OUT}")

    # final test accuracy using the best saved weights.
    model.head.load_state_dict(torch.load(OUT, map_location=device)["state_dict"])
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
