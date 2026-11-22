
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from src.data.dataset import create_dataloaders
from src.models.resemotenet import ResEmoteNet


# ==========================================
# 1. Configuration
# ==========================================

SEED = 42
BATCH_SIZE = 32

CSV_PATH = Path("fer2013/fer2013/fer2013.csv")

CHECKPOINT_PATH = Path(
    "checkpoints/best_resemotenet_aug_dropout.pth"
)

OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ==========================================
# 2. Reproducibility
# ==========================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# 3. Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("ResEmoteNet - Final Evaluation")
print("=" * 55)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 4. Check Files
# ==========================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )


# ==========================================
# 5. Load Dataset
# ==========================================

train_loader, val_loader, test_loader = create_dataloaders(
    csv_path=CSV_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0
)

print("\nDataset loaded successfully")
print("PrivateTest batches:", len(test_loader))


# ==========================================
# 6. Load Best Model
# ==========================================

model = ResEmoteNet(
    num_classes=7,
    in_channels=1,
    dropout_rate=0.5
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nBest checkpoint loaded successfully")

print("Best Epoch:", checkpoint["epoch"])

print(
    "Best Validation Accuracy:",
    f'{checkpoint["best_val_accuracy"]:.2f}%'
)


# ==========================================
# 7. Final Evaluation
# ==========================================

all_predictions = []
all_labels = []

print("\nEvaluating PrivateTest...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# ==========================================
# 8. Calculate Metrics
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )
)


# ==========================================
# 9. Print Results
# ==========================================

print("\n" + "=" * 55)
print("FINAL TEST RESULTS - PrivateTest")
print("=" * 55)

print(f"Accuracy:          {accuracy * 100:.2f}%")

print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1-Score:    {macro_f1:.4f}")

print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print(f"Weighted F1-Score:  {weighted_f1:.4f}")


# ==========================================
# 10. Classification Report
# ==========================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(7)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 55)
print("CLASSIFICATION REPORT")
print("=" * 55)

print(report)


# Save report
report_path = OUTPUT_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("ResEmoteNet - PrivateTest Evaluation\n\n")

    file.write(
        f"Best Epoch: {checkpoint['epoch']}\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}%\n\n"
    )

    file.write(
        f"PrivateTest Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(report)


# ==========================================
# 11. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(7))
)

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest")

plt.title("ResEmoteNet - Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

threshold = cm.max() / 2.0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()

confusion_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. Save Summary
# ==========================================

summary_path = OUTPUT_DIR / "evaluation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("ResEmoteNet Final Evaluation\n")
    file.write("=" * 40 + "\n")

    file.write(f"Best Epoch: {checkpoint['epoch']}\n")

    file.write(
        f"Validation Accuracy: "
        f"{checkpoint['best_val_accuracy']:.2f}%\n"
    )

    file.write(
        f"PrivateTest Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro Precision: {macro_precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: {macro_recall:.4f}\n"
    )

    file.write(
        f"Macro F1-Score: {macro_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1-Score: {weighted_f1:.4f}\n"
    )


# ==========================================
# 13. Completion
# ==========================================

print("\n" + "=" * 55)
print("EVALUATION COMPLETED")
print("=" * 55)

print("Classification Report:", report_path)
print("Confusion Matrix:", confusion_path)
print("Summary:", summary_path)

print("\nPrivateTest was used only for final evaluation.")
