
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import create_dataloaders
from src.models.resemotenet import ResEmoteNet


# ==========================================
# 1. Configuration
# ==========================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

PATIENCE = 10

DROPOUT_RATE = 0.5

CSV_PATH = Path("fer2013/fer2013/fer2013.csv")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Save the new model separately.
BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "best_resemotenet_aug_dropout.pth"
)


# ==========================================
# 2. Reproducibility
# ==========================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# 3. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("ResEmoteNet - Augmentation + Dropout")
print("=" * 50)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 4. Load Dataset
# ==========================================

train_loader, val_loader, test_loader = create_dataloaders(
    csv_path=CSV_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print("\nDataset loaded successfully")

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# ==========================================
# 5. Initialize Model
# ==========================================

model = ResEmoteNet(
    num_classes=7,
    in_channels=1,
    dropout_rate=DROPOUT_RATE
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ==========================================
# 6. Training Function
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 7. Validation Function
# ==========================================

@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 8. Training Loop
# ==========================================

best_val_accuracy = 0.0
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(EPOCHS):

    print("\n" + "=" * 50)
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print("=" * 50)

    # Training
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    # Validation
    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion
    )

    # Learning rate scheduler
    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    # Save history
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    # Print results
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Learning Rate: {current_lr:.7f}")

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "dropout_rate": DROPOUT_RATE,
                "history": history
            },
            BEST_MODEL_PATH
        )

        print("Best model saved!")

    else:

        epochs_without_improvement += 1

        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    # Early stopping
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# ==========================================
# 9. Training Summary
# ==========================================

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

print(f"Best model saved at: {BEST_MODEL_PATH}")

print("\nPrivateTest was not used during training.")
print("It will be used later for final evaluation.")
