import random
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (precision_score, recall_score, f1_score,)
from datasets.cub_cv import CUBDataset
from transforms.transforms import get_transforms
from models.backbone import ResNet18Backbone
from models.prob_cbm import ProbCBM
from models.pem import kl_loss

def main(epochs=100, batch_size=32, SEED=42, train_dataset=None, val_dataset=None, fold_id=None, dataset_name="cub",):
    """
    Train the probabilistic concept predictor.
    """
    # CONFIG
    LEARNING_RATE = 1e-4
    LAMBDA_KL = 5e-5
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

    RESULTS_DIR = (f"results/{dataset_name}/prob_concepts_cv/seed{SEED}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_model_path = os.path.join(RESULTS_DIR, f"prob_concept_predictor_fold{fold_id}_seed{SEED}.pth")

    print(f"{dataset_name.upper()} PROBCBM CONCEPT TRAINING")
    print("Using device:", DEVICE)
    print("Using seed:", SEED)
    print("Saving to:", RESULTS_DIR)
    print("Checkpoint:", best_model_path)

    # DATASETS
    if dataset_name.lower() == "cub":

        SPLIT_ROOT = "data/cub/class_attr_data_10"
        IMAGE_ROOT = "data/cub/CUB_200_2011/images"

        if train_dataset is None:
            train_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="train", transform=get_transforms("train"),)

        if val_dataset is None:
            val_dataset = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="val", transform=get_transforms("val"),)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,)

    backbone = ResNet18Backbone(pretrained=True)
    model = ProbCBM(backbone=backbone, feature_dim=512, num_concepts=train_dataset.num_concepts, num_classes=train_dataset.num_classes,).to(DEVICE)

    concept_criterion = nn.BCELoss()

    # Freezing classifier during concept training
    for p in model.project.parameters():
        p.requires_grad = False

    model.class_anchor.requires_grad = False
    model.beta.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # HISTORY
    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    train_f1s = []
    val_f1s = []

    train_precisions = []
    val_precisions = []

    train_recalls = []
    val_recalls = []

    epoch_times = []

    best_epoch = 0
    best_val_acc = 0
    best_val_f1 = 0
    best_val_precision = 0
    best_val_recall = 0
    
    epochs_without_improvement = 0

    training_start = time.time()

    # TRAINING
    for epoch in range(NUM_EPOCHS):

        epoch_start = time.time()

        model.train()

        train_loss = 0.0
        train_bce = 0.0
        train_kl = 0.0

        train_correct = 0
        train_total = 0

        train_true = []
        train_pred = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for batch in train_bar:

            images = batch["image"].to(DEVICE)
            concepts = batch["concepts"].float().to(DEVICE)

            # Forward
            outputs = model(images)
            concept_probs = outputs["concept_probs"]
            mu = outputs["mu"]
            log_sigma = outputs["log_sigma"]

            # Losses
            bce_loss = concept_criterion(concept_probs, concepts)
            kl = kl_loss(mu, log_sigma)
            loss = bce_loss + LAMBDA_KL * kl

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_bce += bce_loss.item()
            train_kl += kl.item()
            preds = (concept_probs > 0.5)

            train_true.extend(concepts.cpu().numpy().flatten())
            train_pred.extend(preds.cpu().numpy().flatten())

            train_correct += (preds == concepts.bool()).sum().item()
            train_total += concepts.numel()

            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*train_correct/train_total:.2f}%")

        # Epoch metrics
        train_loss_epoch = train_loss / len(train_loader)
        train_bce_epoch = train_bce / len(train_loader)
        train_kl_epoch = train_kl / len(train_loader)
        train_acc = train_correct / train_total

        train_precision = precision_score(train_true, train_pred, zero_division=0)
        train_recall = recall_score(train_true, train_pred, zero_division=0)
        train_f1 = f1_score(train_true, train_pred, zero_division=0)

        train_losses.append(train_loss_epoch)
        train_accs.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        # VALIDATION
        model.eval()

        val_loss = 0.0
        val_bce = 0.0
        val_kl = 0.0

        val_correct = 0
        val_total = 0

        val_true = []
        val_pred = []

        with torch.no_grad():

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")

            for batch in val_bar:
                images = batch["image"].to(DEVICE)
                concepts = batch["concepts"].float().to(DEVICE)

                # Forward
                outputs = model(images)
                concept_probs = outputs["concept_probs"]
                mu = outputs["mu"]
                log_sigma = outputs["log_sigma"]

                # Loss
                bce_loss = concept_criterion(concept_probs, concepts)
                kl = kl_loss(mu, log_sigma)
                loss = bce_loss + LAMBDA_KL * kl

                # Statistics
                val_loss += loss.item()
                val_bce += bce_loss.item()
                val_kl += kl.item()

                preds = (concept_probs > 0.5)

                val_true.extend(concepts.cpu().numpy().flatten())
                val_pred.extend(preds.cpu().numpy().flatten())

                val_correct += (preds == concepts.bool()).sum().item()
                val_total += concepts.numel()

                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*val_correct/val_total:.2f}%")

            # Validation metrics
            val_loss_epoch = val_loss / len(val_loader)
            val_bce_epoch = val_bce / len(val_loader)
            val_kl_epoch = val_kl / len(val_loader)
            val_acc = val_correct / val_total

            val_precision = precision_score(val_true, val_pred, zero_division=0)
            val_recall = recall_score(val_true, val_pred, zero_division=0)
            val_f1 = f1_score(val_true, val_pred, zero_division=0)

            val_losses.append(val_loss_epoch)
            val_accs.append(val_acc)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1s.append(val_f1)

            # Epoch time
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Save best model
            if val_f1 > best_val_f1 + MIN_DELTA:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_val_precision = val_precision
                best_val_recall = val_recall
                epochs_without_improvement = 0

                torch.save(model.state_dict(), best_model_path)

                print(
                    f"New best model saved "
                    f"(Val Acc={100*val_acc:.2f}%)"
                )
            else:
                epochs_without_improvement += 1

            # Print epoch
            print()
            print("="*80)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(
                f"Train Loss={train_loss_epoch:.4f} "
                f"(BCE={train_bce_epoch:.4f}, "
                f"KL={train_kl_epoch:.4f}) "
                f"| Acc={100*train_acc:.2f}%"
            )
            print(
                f"Val Loss={val_loss_epoch:.4f} "
                f"(BCE={val_bce_epoch:.4f}, "
                f"KL={val_kl_epoch:.4f}) "
                f"| Acc={100*val_acc:.2f}%"
            )
            print(
                f"Train F1={train_f1:.4f} "
                f"P={train_precision:.4f} "
                f"R={train_recall:.4f}"
            )
            print(
                f"Val F1={val_f1:.4f} "
                f"P={val_precision:.4f} "
                f"R={val_recall:.4f}"
            )
            print(f"Epochs without improvement: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
            print(f"Time: {epoch_time:.2f}s")
            print("="*80)

        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered!")
            print(
                f"No improvement in validation accuracy "
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    # TRAINING FINISHED
    total_training_time = time.time() - training_start
    training_minutes = int(total_training_time // 60)
    training_seconds = int(total_training_time % 60)

    print()
    print(
        f"Total Training Time: "
        f"{training_minutes} min "
        f"{training_seconds} s"
    )
    print()
    print("Training Finished!")
    print(
        f"Best Validation Accuracy: "
        f"{100*best_val_acc:.2f}%"
    )

    # HISTORY
    history = pd.DataFrame({
        "epoch": range(1, len(train_losses)+1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_precision": train_precisions,
        "val_precision": val_precisions,
        "train_recall": train_recalls,
        "val_recall": val_recalls,
        "train_f1": train_f1s,
        "val_f1": val_f1s,
        "epoch_time_seconds": epoch_times,
    })

    history_path = os.path.join(RESULTS_DIR, f"history_prob_concepts_fold{fold_id}_seed{SEED}.csv")
    history.to_csv(history_path, index=False)

    print(
        f"History saved to "
        f"{history_path}"
    )

    # SUMMARY FILE
    summary_path = os.path.join(RESULTS_DIR, f"training_summary_prob_concepts_fold{fold_id}_seed{SEED}.txt")

    with open(summary_path, "w") as f:
        f.write("BEST MODEL\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f}\n\n")
        f.write(f"Validation Precision: {val_precision:.4f}\n")
        f.write(f"Best Validation Precision: {best_val_precision:.4f}\n")
        f.write(f"Best Validation Recall: {best_val_recall:.4f}\n")
        f.write(f"Validation Recall: {val_recall:.4f}\n")
        f.write(f"Validation F1: {val_f1:.4f}\n\n")
        f.write(f"Final BCE Loss: {val_bce_epoch:.6f}\n")
        f.write(f"Final KL Loss: {val_kl_epoch:.6f}\n")
        f.write(f"Final Total Loss: {val_loss_epoch:.6f}\n\n")
        f.write(f"Training Time: {total_training_time/60:.2f} minutes\n")

    # RETURN METRICS
    return {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_c_acc": best_val_acc,
        "last_val_c_precision": val_precision,
        "last_val_c_recall": val_recall,
        "best_val_c_precision": best_val_precision,
        "best_val_c_recall": best_val_recall,
        "last_val_c_f1": val_f1,
        "best_val_c_f1": best_val_f1,
        "fold_time_minutes": total_training_time / 60,
    }