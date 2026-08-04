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

def main(epochs=100, batch_size=32, SEED=42, train_dataset=None, val_dataset=None, fold_id=None, dataset_name="cub"):
    """
    Train the sequential CBM classifier.
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

    RESULTS_DIR = f"results/{dataset_name}/sequential_cv/seed{SEED}"
    BATCH_SIZE = batch_size
    NUM_EPOCHS = epochs
    
    print(f"{dataset_name.upper()} SEQUENTIAL_CV TRAINING")
    print("Using device: ", DEVICE)
    print("Using seed: ", SEED)

    if dataset_name.lower() == "cub":
        SPLIT_ROOT = "data/cub/class_attr_data_10"
        IMAGE_ROOT = "data/cub/CUB_200_2011/images"
        # DATASETS
        if train_dataset is None:
            train_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="train", transform=get_transforms("train"))
        if val_dataset is None:
            val_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="val", transform=get_transforms("val"))

    train_y_losses = []
    val_y_losses = []

    train_y_accs = []
    val_y_accs = []

    best_val_y_acc = 0.0
    best_val_y_loss = 0.0
    best_val_y_precision = 0.0
    best_val_y_recall = 0.0
    best_val_y_f1 = 0.0

    best_train_y_acc = 0.0
    best_train_y_precision = 0.0
    best_train_y_recall = 0.0
    best_train_y_f1 = 0.0

    best_epoch = 0
    epochs_without_improvement = 0
    epoch_times = []

    train_y_precisions = []
    train_y_recalls = []
    train_y_f1s = []
    
    val_y_precisions = []
    val_y_recalls = []
    val_y_f1s = []
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_model_path = os.path.join(RESULTS_DIR, f"sequential_cbm_fold{fold_id}_seed{SEED}.pth")
    print(f"Sequential model: {best_model_path}")
    
    # DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # MODEL
    backbone = ResNet18Backbone(pretrained=True)
    model = CBM(backbone=backbone, num_concepts=train_dataset.num_concepts, num_classes=train_dataset.num_classes,).to(DEVICE)

    # LOAD PRETRAINED CONCEPT PREDICTOR - STAGE ONE
    concept_model_path = (
        f"results/{dataset_name}/concepts_cv/seed{SEED}/"
        f"concept_predictor_fold{fold_id}_seed{SEED}.pth"
    )
    model.load_state_dict(torch.load(concept_model_path, map_location=DEVICE), strict=False)
    #model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "concept_predictor.pth"), map_location=DEVICE))
    print(f"Concept model: {concept_model_path}")

    # FREEZE BACKBONE
    for p in model.backbone.parameters():
        p.requires_grad = False
    # FREEZE CONCEPT HEAD
    for p in model.concept_head.parameters():
        p.requires_grad = False
    # UNFREEZE TASK HEAD
    for p in model.task_head.parameters():
        p.requires_grad = True

    # LOSSES
    task_criterion = nn.CrossEntropyLoss()

    # OPTIMIZER
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    # TRAINING
    training_start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_y_true = []
        train_y_pred = []

        epoch_start = time.time()
        model.train()

        train_y_loss = 0.0
        train_y_correct = 0
        train_y_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=True)
        for batch in train_bar:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Forward pass
            outputs = model(images)
            y_logits = outputs["class_logits"]

            # Loss
            y_loss = task_criterion(y_logits, labels)
            
            # Optimization
            optimizer.zero_grad()
            y_loss.backward()
            optimizer.step()

            # Running losses
            train_y_loss += y_loss.item()
            
            # Predictions
            y_pred = y_logits.argmax(dim=1)

            # store the predictions and true labels for metrics
            train_y_true.extend(labels.cpu().numpy())

            train_y_pred.extend(y_pred.cpu().numpy())

            # Running Accuracies
            train_y_correct += (y_pred == labels).sum().item()
            train_y_total += labels.size(0)
            
            # Progress bar
            train_bar.set_postfix(
                loss=f"{y_loss.item():.4f}",
                y_acc=f"{100*train_y_correct/train_y_total:.2f}%",
                )

        # Epochs metrics
        train_y_loss_epoch = train_y_loss / len(train_loader)
        train_y_acc = train_y_correct / train_y_total
        train_y_precision = precision_score(train_y_true, train_y_pred, average="macro", zero_division=0)
        train_y_recall = recall_score(train_y_true, train_y_pred, average="macro", zero_division=0)
        train_y_f1 = f1_score(train_y_true, train_y_pred, average="macro", zero_division=0)
        
        # Append metrics
        train_y_losses.append(train_y_loss_epoch)
        train_y_accs.append(train_y_acc)
        train_y_precisions.append(train_y_precision)
        train_y_recalls.append(train_y_recall)
        train_y_f1s.append(train_y_f1)

        # VALIDATION
        model.eval()

        val_y_loss = 0.0
        val_y_correct = 0
        val_y_total = 0

        val_y_true = []
        val_y_pred = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=True, dynamic_ncols=True)

            for batch in val_bar:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                y_logits = outputs["class_logits"]

                # Compute Losses
                y_loss = task_criterion(y_logits, labels)

                # Running losses
                val_y_loss += y_loss.item()

                # Predictions
                y_pred = y_logits.argmax(dim=1)

                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(y_pred.cpu().numpy())

                # Running Acc
                val_y_correct += (y_pred == labels).sum().item()
                val_y_total += labels.size(0)

                # Progress bar
                val_bar.set_postfix(
                    loss=f"{y_loss.item():.4f}",
                    y_acc=f"{100*val_y_correct/val_y_total:.2f}%",
                    )

        # Compute Val epoch metrics        
        val_y_loss_epoch = val_y_loss / len(val_loader)
        val_y_acc = val_y_correct / val_y_total
        val_y_precision = precision_score(val_y_true, val_y_pred, average="macro", zero_division=0)
        val_y_recall = recall_score(val_y_true, val_y_pred, average="macro", zero_division=0)
        val_y_f1 = f1_score(val_y_true, val_y_pred, average="macro", zero_division=0)

        # append metrics
        val_y_losses.append(val_y_loss_epoch)
        val_y_accs.append(val_y_acc)
        val_y_precisions.append(val_y_precision)
        val_y_recalls.append(val_y_recall)
        val_y_f1s.append(val_y_f1)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # SAVE BEST MODEL
        if val_y_acc > best_val_y_acc + MIN_DELTA:

            best_val_y_acc = val_y_acc
            best_epoch = epoch + 1

            best_val_y_loss = val_y_loss_epoch
            best_val_y_precision = val_y_precision
            best_val_y_recall = val_y_recall
            best_val_y_f1 = val_y_f1

            best_train_y_acc = train_y_acc
            best_train_y_precision = train_y_precision
            best_train_y_recall = train_y_recall
            best_train_y_f1 = train_y_f1

            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (Val Acc = {val_y_acc:.4f})")
        else:
            epochs_without_improvement += 1

        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(
            f"Train -> "
            f"Loss: {train_y_loss_epoch:.4f} | "
            f"Acc: Y={100*train_y_acc:.2f}% |"
            f"F1={train_y_f1:.4f} |"
            f"P={train_y_precision:.4f} |"
            f"R={train_y_recall:.4f} |"
        )
        print(
            f"Val   -> "
            f"Loss: {val_y_loss_epoch:.4f} | "
            f"Acc: Y={100*val_y_acc:.2f}% |"
            f"F1={val_y_f1:.4f} |"
            f"P={val_y_precision:.4f} |"
            f"R={val_y_recall:.4f} |"
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
    training_minutes = int(total_training_time // 60)
    training_seconds = int(total_training_time % 60)

    print(
        f"\nTotal Training Time: "
        f"{training_minutes} min {training_seconds} s"
    )

    print("\nTraining Finished!")
    print(f"Best Validation Accuracy: {best_val_y_acc:.4f}")

    history = pd.DataFrame({
        "epoch": range(1, len(train_y_losses) + 1),
        "train_y_loss": train_y_losses,
        "val_y_loss": val_y_losses,
        "train_y_acc": train_y_accs,
        "val_y_acc": val_y_accs,
        "epoch_time_seconds": epoch_times,
        "train_y_precision": train_y_precisions,
        "val_y_precision": val_y_precisions,
        "train_y_recall": train_y_recalls,
        "val_y_recall": val_y_recalls,
        "train_y_f1": train_y_f1s,
        "val_y_f1": val_y_f1s,
    })

    history_path = os.path.join(RESULTS_DIR, f"history_sequential_fold{fold_id}_seed{SEED}.csv")
    history.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    results_file = os.path.join(RESULTS_DIR, f"training_summary_sequential_fold{fold_id}_seed{SEED}.txt")

    with open(results_file, "w") as f:
        f.write("BEST MODEL\n")
        f.write(f"Epoch: {best_epoch}\n")
        f.write(f"Validation Y Accuracy: {best_val_y_acc:.4f}\n")
        f.write(f"Validation Y Precision: {best_val_y_precision:.4f}\n")
        f.write(f"Validation Y Recall: {best_val_y_recall:.4f}\n")
        f.write(f"Validation Y F1: {best_val_y_f1:.4f}\n")
        f.write(f"Validation Y Loss: {best_val_y_loss:.4f}\n")

        f.write("\nTrain Metrics (Best Epoch)\n")
        f.write(f"Train Y Accuracy: {best_train_y_acc:.4f}\n")
        f.write(f"Train Y Precision: {best_train_y_precision:.4f}\n")
        f.write(f"Train Y Recall: {best_train_y_recall:.4f}\n")
        f.write(f"Train Y F1: {best_train_y_f1:.4f}\n")

        f.write("FINAL EPOCH\n")
        f.write(f"Validation Y Accuracy: {val_y_acc:.4f}\n")
        f.write(f"Validation Y Precision: {val_y_precision:.4f}\n")
        f.write(f"Validation Y Recall: {val_y_recall:.4f}\n")
        f.write(f"Validation Y F1: {val_y_f1:.4f}\n")
        f.write(f"Validation Y Loss: {val_y_loss_epoch:.4f}\n")

        f.write(f"Total Training Time: {total_training_time/60:.2f} minutes\n")

    return {
        "fold": fold_id,
        "best_epoch": best_epoch,

        "best_val_y_acc": best_val_y_acc,
        "best_val_y_precision": best_val_y_precision,
        "best_val_y_recall": best_val_y_recall,
        "best_val_y_f1": best_val_y_f1,

        "fold_time_minutes": total_training_time / 60
    }
if __name__ == "__main__":
    main()