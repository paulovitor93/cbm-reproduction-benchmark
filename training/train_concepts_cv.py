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
from models.cbm import CBM
from models.backbone import ResNet18Backbone
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(epochs=100, batch_size=32, SEED=42, train_dataset=None, val_dataset=None, fold_id=None, dataset_name="cub"):
    """
    Train the concept predictor.
    """
    # CONFIG
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 1e-4

    RESULTS_DIR = f"results/{dataset_name}/concepts_cv/seed{SEED}"
    BATCH_SIZE = batch_size
    NUM_EPOCHS = epochs

    print(f"{dataset_name.upper()} CONCEPT_CV TRAINING")    
    print("Using device: ", DEVICE)
    print(f"Results dir: {RESULTS_DIR}")
    print("Using seed: ", SEED)
    
    if dataset_name.lower() == "cub":
        SPLIT_ROOT = "data/cub/class_attr_data_10"
        IMAGE_ROOT = "data/cub/CUB_200_2011/images"

        # DATASETS
        if train_dataset is None:
            train_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="train", transform=get_transforms("train"))
        if val_dataset is None:
            val_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="val", transform=get_transforms("val"))

    train_c_losses = []
    val_c_losses = []

    train_c_accs = []
    val_c_accs = []

    best_val_c_f1 = 0.0
    best_val_c_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_times = []

    train_c_precisions = []
    train_c_recalls = []
    train_c_f1s = []

    val_c_precisions = []
    val_c_recalls = []
    val_c_f1s = []
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_model_path = os.path.join(RESULTS_DIR, f"concept_predictor_fold{fold_id}_seed{SEED}.pth")
    print(f"Checkpoint: {best_model_path}")

    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

    # MODEL
    backbone = ResNet18Backbone(pretrained=True)
    model = CBM(backbone=backbone, num_concepts=train_dataset.num_concepts, num_classes=train_dataset.num_classes,).to(DEVICE)

    # LOSSES
    concept_criterion = nn.BCEWithLogitsLoss()
    
    # FREEZE TASK HEAD
    for p in model.task_head.parameters():
        p.requires_grad = False 

    # OPTIMIZER
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    # TRAINING
    training_start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_c_true = []
        train_c_pred = []

        epoch_start = time.time()
        model.train()

        train_c_loss = 0.0
        train_c_correct = 0
        train_c_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=True)
        for batch in train_bar:
            images = batch["image"].to(DEVICE)
            concepts = batch["concepts"].to(DEVICE)

            # Forward pass
            outputs = model(images)
            c_logits = outputs["concept_logits"]

            # Loss
            c_loss = concept_criterion(c_logits, concepts)
            
            # Optimization
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

            # Running losses
            train_c_loss += c_loss.item()
            
            # Predictions
            c_pred = (torch.sigmoid(c_logits) > 0.5)

            # store the predictions and true labels for metrics
            train_c_true.extend( concepts.cpu().numpy().flatten())
            train_c_pred.extend(c_pred.cpu().numpy().flatten())

            # Running Accuracies
            train_c_correct += (c_pred == concepts.bool()).sum().item()
            train_c_total += concepts.numel()
            
            # Progress bar
            train_bar.set_postfix(
                loss=f"{c_loss.item():.4f}",
                c_acc=f"{100*train_c_correct/train_c_total:.2f}%"
                )

        # Epochs metrics
        train_c_loss_epoch = train_c_loss / len(train_loader)
        train_c_acc = train_c_correct / train_c_total
        train_c_precision = precision_score(train_c_true, train_c_pred, zero_division=0)
        train_c_recall = recall_score(train_c_true, train_c_pred, zero_division=0)
        train_c_f1 = f1_score(train_c_true, train_c_pred, zero_division=0)
        
        # Append metrics
        train_c_losses.append(train_c_loss_epoch)
        train_c_accs.append(train_c_acc)
        train_c_precisions.append(train_c_precision)
        train_c_recalls.append(train_c_recall)
        train_c_f1s.append(train_c_f1)

        # VALIDATION
        model.eval()

        val_c_loss = 0.0
        val_c_correct = 0
        val_c_total = 0
        val_c_true = []
        val_c_pred = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=True, dynamic_ncols=True)

            for batch in val_bar:
                images = batch["image"].to(DEVICE)
                concepts = batch["concepts"].to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                c_logits = outputs["concept_logits"]

                # Compute Losses
                c_loss = concept_criterion(c_logits, concepts)

                # Running losses
                val_c_loss += c_loss.item()

                # Predictions
                c_pred = (torch.sigmoid(c_logits) > 0.5)

                val_c_true.extend(concepts.cpu().numpy().flatten())
                val_c_pred.extend(c_pred.cpu().numpy().flatten())

                # Running Acc
                val_c_correct += (c_pred == concepts.bool()).sum().item()
                val_c_total += concepts.numel()

                # Progress bar
                val_bar.set_postfix(loss=f"{c_loss.item():.4f}", c_acc=f"{100*val_c_correct/val_c_total:.2f}%")

        # Compute Val epoch metrics        
        val_c_loss_epoch = val_c_loss / len(val_loader)
        val_c_acc = (val_c_correct / val_c_total)
        val_c_precision = precision_score(val_c_true, val_c_pred, zero_division=0)
        val_c_recall = recall_score(val_c_true, val_c_pred, zero_division=0)
        val_c_f1 = f1_score(val_c_true, val_c_pred, zero_division=0)

        # append metrics
        val_c_losses.append(val_c_loss_epoch)
        val_c_accs.append(val_c_acc)

        val_c_precisions.append(val_c_precision)
        val_c_recalls.append(val_c_recall)
        val_c_f1s.append(val_c_f1)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # SAVE BEST MODEL
        if val_c_f1 > best_val_c_f1 + MIN_DELTA:

            best_val_c_f1 = val_c_f1
            best_val_c_acc = val_c_acc
            best_epoch = epoch + 1

            best_val_c_precision = val_c_precision
            best_val_c_recall = val_c_recall

            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (Val F1 = {val_c_f1:.4f})")
        else:
            epochs_without_improvement += 1

        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(
            f"Train -> "
            f"Loss: {train_c_loss_epoch:.4f} "
            f"Acc: C={100*train_c_acc:.2f}%"
        )
        print(
            f"Val   -> "
            f"Loss: {val_c_loss_epoch:.4f} "
            f"Acc: C={100*val_c_acc:.2f}%"
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
                f"No improvement in validation F1 "
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    total_training_time = time.time() - training_start
    training_minutes = int(total_training_time // 60)
    training_seconds = int(total_training_time % 60)

    print(
        f"\nTotal Training Time: "
        f"{training_minutes} min {training_seconds} s"
    )

    print("\nTraining Finished!")
    print(f"Best Validation Concept F1: {best_val_c_f1:.4f}")

    history = pd.DataFrame({
        "epoch": range(1, len(train_c_losses) + 1),
        "train_c_loss": train_c_losses,
        "val_c_loss": val_c_losses,
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

    history_path = os.path.join(RESULTS_DIR, f"history_concepts_cv_fold{fold_id}_seed{SEED}.csv")

    history.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    results_file = os.path.join(RESULTS_DIR, f"training_summary_concepts_cv_fold{fold_id}_seed{SEED}.txt")

    with open(results_file, "w") as f:
        f.write("BEST MODEL\n")
        f.write(f"Epoch: {best_epoch}\n")
        f.write(f"Best Validation Concept Accuracy: {best_val_c_acc:.4f}\n\n")
        f.write(f"Best Validation Concept Loss: {val_c_losses[best_epoch-1]:.4f}\n")

        f.write(f"Best Validation Concept Precision: {best_val_c_precision:.4f}\n")
        f.write(f"Best Validation Concept Recall: {best_val_c_recall:.4f}\n")
        f.write(f"Best Validation Concept F1: {best_val_c_f1:.4f}\n\n")

        f.write(f"Train Concept Accuracy: {train_c_accs[best_epoch-1]:.4f}\n")
        f.write(f"Train Concept F1: {train_c_f1s[best_epoch-1]:.4f}\n")
        f.write(f"Train Concept Precision: {train_c_precisions[best_epoch-1]:.4f}\n")
        f.write(f"Train Concept Recall: {train_c_recalls[best_epoch-1]:.4f}\n\n")

        f.write("FINAL EPOCH\n")
        f.write(f"Validation C Accuracy: {val_c_acc:.4f}\n\n")

        f.write(f"Total Training Time: {total_training_time/60:.2f} minutes\n")
    
    return {
        "fold": fold_id,
        "best_epoch": best_epoch,

        "best_val_c_acc": best_val_c_acc,

        "best_val_c_precision": best_val_c_precision,
        "best_val_c_recall": best_val_c_recall,
        "best_val_c_f1": best_val_c_f1,

        "fold_time_minutes": total_training_time / 60
    }

if __name__ == "__main__":
    main()