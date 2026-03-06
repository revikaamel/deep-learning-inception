# inception_intel.py
"""
Inception v3 Deep Learning Project - Intel Image Classification
Output:
- Grafik: training_curves.png
- Confusion Matrix: confusion_matrix.png
- Report evaluasi: report_inception.txt
"""

import os
import time
import copy
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from torchvision.models import Inception_V3_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== CONFIG ======================
DATA_DIR_TRAIN = "data/seg_train"
DATA_DIR_TEST = "data/seg_test"
BATCH_SIZE = 8  # kecilkan agar aman untuk RAM 4GB
NUM_EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows + CPU
SEED = 42
MODEL_SAVE_PATH = "inception_best.pth"
USE_GPU = torch.cuda.is_available()
# ====================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if USE_GPU else "cpu")
print("Device:", device)

# ================== METRICS ==================
def compute_metrics_all(y_true, y_pred, y_prob, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_true_bin, y_prob, average=average, multi_class='ovr')
    except:
        roc_auc = float('nan')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}

# ================== DATA PREPARATION ==================
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=20,
        scale=(0.8, 1.0),
        shear=20
    ),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load train dataset
full_train_dataset = datasets.ImageFolder(DATA_DIR_TRAIN, transform=train_transforms)
num_train_total = len(full_train_dataset)
indices = list(range(num_train_total))
random.shuffle(indices)

# Split train/val
n_val = int(0.2 * num_train_total)
train_indices = indices[n_val:]
val_indices = indices[:n_val]

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(datasets.ImageFolder(DATA_DIR_TRAIN, transform=val_transforms), val_indices)

# Load test dataset
test_dataset = datasets.ImageFolder(DATA_DIR_TEST, transform=val_transforms)

class_names = full_train_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ================== MODEL ==================
model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
if model.aux_logits:
    aux_in = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ================== TRAIN & EVAL FUNCTIONS ==================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, preds_all, targets_all, probs_all = 0, [], [], []
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, aux_outputs = model(inputs)
        loss = criterion(outputs, targets)
        if aux_outputs is not None:
            loss += 0.4 * criterion(aux_outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs.detach(), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        preds_all.extend(preds.tolist())
        targets_all.extend(targets.cpu().numpy().tolist())
        probs_all.extend(probs.tolist())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics_all(targets_all, preds_all, np.array(probs_all))
    return epoch_loss, metrics

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, preds_all, targets_all, probs_all = 0, [], [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            preds_all.extend(preds.tolist())
            targets_all.extend(targets.cpu().numpy().tolist())
            probs_all.extend(probs.tolist())
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics_all(targets_all, preds_all, np.array(probs_all))
    return epoch_loss, metrics

# ================== MAIN ==================
if __name__ == "__main__":
    best_f1 = 0
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())


    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics['f1'])
        t_epoch = time.time() - t0

        # Simpan history
        train_acc_hist.append(train_metrics['accuracy'])
        val_acc_hist.append(val_metrics['accuracy'])
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"Epoch Time: {t_epoch:.1f}s")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved (F1={best_f1:.4f})")

    total_time = time.time() - since
    print(f"\nTraining complete in {total_time:.1f}s | Best F1={best_f1:.4f}")

    # ================== TESTING ==================
    model.load_state_dict(best_model_wts)
    t0_test = time.time()
    test_loss, test_metrics = evaluate(model, test_loader, criterion)
    test_time = time.time() - t0_test
    print(f"\nTesting complete in {test_time:.1f}s")
    print("\n=== FINAL TEST RESULTS ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # ================== PLOT ACCURACY & LOSS ==================
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_hist, label='Train Accuracy')
    plt.plot(val_acc_hist, label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    # ================== CONFUSION MATRIX ==================
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing Confusion Matrix"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Inception V3")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved as confusion_matrix.png")

    # ================== SAVE REPORT ==================
    with open("report_inception.txt", "w") as f:
        f.write("Inception v3 Evaluation Report\n")
        f.write(f"Total Training Time: {total_time:.2f} seconds\n")
        f.write(f"Testing Time: {test_time:.2f} seconds\n")
        f.write("Final Test Metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print("Report saved as report_inception.txt")
