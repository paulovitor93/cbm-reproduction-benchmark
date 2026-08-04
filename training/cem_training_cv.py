import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score)
import pandas as pd
import numpy as np
import time
import os

from datasets.cub_cv import CUBDataset
from transforms.transforms import get_transforms
from models.cem import CEM
from models.backbone import ResNet18Backbone

def main(epochs=100, batch_size=32, SEED=42, train_dataset=None, val_dataset=None, fold_id=None, dataset_name="cub"):
    """
    Train the CEM model.
    """

    # CONFIG
    LEARNING_RATE = 1e-4
    LAMBDA_C = 1.0
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 1e-4

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = batch_size
    NUM_EPOCHS = epochs

    if dataset_name.lower() == "cub":
        SPLIT_ROOT = "data/cub/class_attr_data_10"
        IMAGE_ROOT = "data/cub/CUB_200_2011/images"

        # DATASETS
        if train_dataset is None:
            train_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="train", transform=get_transforms("train"))
        if val_dataset is None:
            val_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="val", transform=get_transforms("val"))

    RESULTS_DIR = f"results/{dataset_name}/cem_cv/seed{SEED}"

    print(f"{dataset_name.upper()} CEM TRAINING")
    print("Using device: ", DEVICE)
    print("Using seed: ", SEED)
    print("Saving to:", RESULTS_DIR)

    train_c_losses = []
    val_c_losses = []

    train_y_losses = []
    val_y_losses = []

    train_c_accs = []
    train_y_accs = []

    val_c_accs = []
    val_y_accs = []

    total_train_losses = []
    total_val_losses = []

    epoch_times = []
    best_val_y_acc = 0.0
    best_val_c_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    train_c_precisions = []
    train_c_recalls = []
    train_c_f1s = []

    val_c_precisions = []
    val_c_recalls = []
    val_c_f1s = []

    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_model_path = os.path.join(RESULTS_DIR, f"best_model_fold{fold_id}_seed{SEED}.pth")
    print("Model path:", best_model_path)

    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

    # MODEL
    backbone = ResNet18Backbone(pretrained=True)
    model = CEM(backbone=backbone, num_concepts=train_dataset.num_concepts, num_classes=train_dataset.num_classes,).to(DEVICE)

    # LOSSES
    concept_criterion = nn.BCEWithLogitsLoss()
    task_criterion = nn.CrossEntropyLoss()

    # OPTIMIZER
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    # TRAINING
    training_start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_c_true = []
        train_y_true = []

        train_c_pred = []
        train_y_pred = []

        epoch_start = time.time()
        model.train()

        train_c_loss = 0.0
        train_y_loss = 0.0
        train_total_loss = 0.0

        train_y_correct = 0
        train_c_correct = 0

        train_c_total = 0
        train_y_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=True)

        for batch in train_bar:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            concepts = batch["concepts"].to(DEVICE)

            # Forward pass
            outputs = model(images)
            c_logits = outputs["concept_logits"]
            c_probs = outputs["concept_probs"]
            y_logits = outputs["class_logits"]

            # Loss
            c_loss = concept_criterion(c_logits, concepts)
            y_loss = task_criterion(y_logits, labels)
            loss = y_loss + LAMBDA_C * c_loss

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running losses
            train_c_loss += c_loss.item()
            train_y_loss += y_loss.item()
            train_total_loss += loss.item()

            # Predictions
            c_pred = (c_probs > 0.5)
            y_pred = y_logits.argmax(dim=1)

            # store the predictions and true labels for metrics
            train_c_true.extend( concepts.cpu().numpy().flatten())
            train_y_true.extend(labels.cpu().numpy())

            train_c_pred.extend(c_pred.cpu().numpy().flatten())
            train_y_pred.extend(y_pred.cpu().numpy())

            # Running Accuracies
            train_c_correct += (c_pred == concepts.bool()).sum().item()
            train_y_correct += (y_pred == labels).sum().item()

            train_c_total += concepts.numel()
            train_y_total += labels.size(0)

            # Progress bar
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                y_acc=f"{100*train_y_correct/train_y_total:.2f}%",
                c_acc=f"{100*train_c_correct/train_c_total:.2f}%"
                )

        # Epochs metrics
        train_c_loss_epoch = train_c_loss / len(train_loader)
        train_y_loss_epoch = train_y_loss / len(train_loader)
        train_total_loss_epoch = train_total_loss / len(train_loader)

        train_c_acc = train_c_correct / train_c_total
        train_y_acc = train_y_correct / train_y_total

        train_c_precision = precision_score(train_c_true, train_c_pred, zero_division=0)
        train_c_recall = recall_score(train_c_true, train_c_pred, zero_division=0)
        train_c_f1 = f1_score(train_c_true, train_c_pred, zero_division=0)

        # Append metrics
        train_c_losses.append(train_c_loss_epoch)
        train_y_losses.append(train_y_loss_epoch)
        total_train_losses.append(train_total_loss_epoch)
        train_y_accs.append(train_y_acc)
        train_c_accs.append(train_c_acc)
        train_c_precisions.append(train_c_precision)
        train_c_recalls.append(train_c_recall)
        train_c_f1s.append(train_c_f1)

        # VALIDATION
        model.eval()

        val_c_loss = 0.0
        val_y_loss = 0.0
        val_total_loss = 0.0

        val_c_correct = 0
        val_y_correct = 0

        val_c_total = 0
        val_y_total = 0

        val_c_true = []
        val_y_true = []

        val_c_pred = []
        val_y_pred = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=True, dynamic_ncols=True)

            for batch in val_bar:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                concepts = batch["concepts"].to(DEVICE)

                # Forward pass
                outputs = model(images)
                c_logits = outputs["concept_logits"]
                c_probs = outputs["concept_probs"]
                y_logits = outputs["class_logits"]

                # Compute Losses
                c_loss = concept_criterion(c_logits, concepts)
                y_loss = task_criterion(y_logits, labels)
                loss = y_loss + LAMBDA_C * c_loss

                # Running losses
                val_total_loss += loss.item()
                val_c_loss += c_loss.item()
                val_y_loss += y_loss.item()

                # Predictions
                c_pred = (c_probs > 0.5)
                y_pred = y_logits.argmax(dim=1)

                val_c_true.extend(concepts.cpu().numpy().flatten())
                val_y_true.extend(labels.cpu().numpy())

                val_c_pred.extend(c_pred.cpu().numpy().flatten())
                val_y_pred.extend(y_pred.cpu().numpy())

                # Running Acc
                val_y_correct += (y_pred == labels).sum().item()
                val_y_total += labels.size(0)

                val_c_correct += (c_pred == concepts.bool()).sum().item()
                val_c_total += concepts.numel()

                # Progress bar
                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    y_acc=f"{100*val_y_correct/val_y_total:.2f}%",
                    c_acc=f"{100*val_c_correct/val_c_total:.2f}%"
                    )

        # Compute Val epoch metrics
        val_c_loss_epoch = val_c_loss / len(val_loader)
        val_y_loss_epoch = val_y_loss / len(val_loader)
        val_total_loss_epoch = val_total_loss / len(val_loader)

        val_c_acc = (val_c_correct / val_c_total)
        val_y_acc = val_y_correct / val_y_total

        val_c_precision = precision_score(val_c_true, val_c_pred, zero_division=0)
        val_c_recall = recall_score(val_c_true, val_c_pred, zero_division=0)
        val_c_f1 = f1_score(val_c_true, val_c_pred, zero_division=0)

        # append metrics
        val_c_losses.append(val_c_loss_epoch)
        val_y_losses.append(val_y_loss_epoch)
        total_val_losses.append(val_total_loss_epoch)
        val_c_accs.append(val_c_acc)
        val_y_accs.append(val_y_acc)

        val_c_precisions.append(val_c_precision)
        val_c_recalls.append(val_c_recall)
        val_c_f1s.append(val_c_f1)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # SAVE BEST MODEL
        if val_y_acc > best_val_y_acc + MIN_DELTA:

            best_val_y_acc = val_y_acc
            best_val_c_acc = val_c_acc
            best_epoch = epoch + 1

            best_val_c_f1 = val_c_f1
            best_val_c_precision = val_c_precision
            best_val_c_recall = val_c_recall

            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (Val Acc = {val_y_acc:.4f})")
        else:
            epochs_without_improvement += 1

        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(
            f"Train -> "
            f"Loss: {train_total_loss_epoch:.4f} "
            f"(C: {train_c_loss_epoch:.4f}, Y: {train_y_loss_epoch:.4f}) | "
            f"Acc: Y={100*train_y_acc:.2f}% C={100*train_c_acc:.2f}%"
        )
        print(
            f"Val   -> "
            f"Loss: {val_total_loss_epoch:.4f} "
            f"(C: {val_c_loss_epoch:.4f}, Y: {val_y_loss_epoch:.4f}) | "
            f"Acc: Y={100*val_y_acc:.2f}% C={100*val_c_acc:.2f}%"
        )
        print(
            f"Train C -> "
            f"F1={train_c_f1:.4f} "
            f"P={train_c_precision:.4f} "
            f"R={train_c_recall:.4f}"
        )
        print(
            f"Val C -> "
            f"F1={val_c_f1:.4f} "
            f"P={val_c_precision:.4f} "
            f"R={val_c_recall:.4f}"
        )
        print(f"Epochs without improvement: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
        print(f"Time  | {epoch_time:.2f}s")
        print(f"{'='*80}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered!")
            print(
                f"No improvement in validation accuracy "
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    total_training_time = time.time() - training_start
    epochs_ran = len(total_train_losses)
    training_minutes = int(total_training_time // 60)
    training_seconds = int(total_training_time % 60)

    print(
        f"\nTotal Training Time: "
        f"{training_minutes} min {training_seconds} s"
    )

    print("\nTraining Finished!")
    print(f"Best Validation Accuracy: {best_val_y_acc:.4f}")

    history = pd.DataFrame({
        "epoch": range(1, epochs_ran + 1),
        "train_loss": total_train_losses,
        "val_loss": total_val_losses,
        "train_c_loss": train_c_losses,
        "val_c_loss": val_c_losses,
        "train_y_loss": train_y_losses,
        "val_y_loss": val_y_losses,
        "train_y_acc": train_y_accs,
        "val_y_acc": val_y_accs,
        "train_c_acc": train_c_accs,
        "val_c_acc": val_c_accs,
        "train_c_precision": train_c_precisions,
        "val_c_precision": val_c_precisions,
        "train_c_recall": train_c_recalls,
        "val_c_recall": val_c_recalls,
        "train_c_f1": train_c_f1s,
        "val_c_f1": val_c_f1s,
        "epoch_time_seconds": epoch_times,
    })

    history_path = os.path.join(RESULTS_DIR, f"history_cem_cv_fold{fold_id}_seed{SEED}.csv")
    history.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    results_file = os.path.join(RESULTS_DIR, f"training_summary_cem_cv_fold{fold_id}_seed{SEED}.txt")

    with open(results_file, "w") as f:
        f.write(f"Epochs Run: {epochs_ran}\n")
        f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")

        f.write("BEST MODEL\n")
        f.write(f"Epoch: {best_epoch}\n")
        f.write(f"Validation Y Accuracy: {best_val_y_acc:.4f}\n")
        f.write(f"Validation C Accuracy: {best_val_c_acc:.4f}\n\n")

        f.write(f"Validation C Precision: {best_val_c_precision:.4f}\n")
        f.write(f"Validation C Recall: {best_val_c_recall:.4f}\n")
        f.write(f"Validation C F1: {best_val_c_f1:.4f}\n")

        f.write("FINAL EPOCH\n")
        f.write(f"Validation Y Accuracy: {val_y_acc:.4f}\n")
        f.write(f"Validation C Accuracy: {val_c_acc:.4f}\n\n")

        f.write(f"Total Training Time: {total_training_time/60:.2f} minutes\n")

    return {
    "fold": fold_id,
    "best_epoch": best_epoch,

    "best_val_y_acc": best_val_y_acc,

    "best_val_c_acc": best_val_c_acc,
    "best_val_c_precision": best_val_c_precision,
    "best_val_c_recall": best_val_c_recall,
    "best_val_c_f1": best_val_c_f1,

    "fold_time_minutes": total_training_time / 60
    }

if __name__ == "__main__":
    main()