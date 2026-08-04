import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os

from sklearn.metrics import (precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets.cub_cv import CUBDataset
from datasets.awa2 import AwA2Dataset
from datasets.derm7pt import Derm7ptDataset
from datasets.celeba import CelebADataset
from datasets.synthetic import SyntheticDataset
from training.cbm_cross_validation import TransformSubset
from transforms.transforms import get_transforms
from models.backbone import ResNet18Backbone
from models.prob_cbm import ProbCBM

def evaluate_fold(model_path, test_loader, device, fold):
    """
    Evaluate one trained ProbCBM concept predictor on the test set.
    """

    # Model setup
    backbone = ResNet18Backbone(pretrained=True)
    model = ProbCBM(backbone=backbone, feature_dim=backbone.feature_dim, num_concepts=test_loader.dataset.num_concepts,
                    num_classes=test_loader.dataset.num_classes, mc_samples=50,).to(device)
    
    assert os.path.exists(model_path), (f"Checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loading model at: {model_path}")

    model.eval()

    # Criterions
    concept_criterion = nn.BCELoss()

    test_c_loss = 0.0

    test_c_correct = 0

    test_c_total = 0

    test_c_true = []
    test_c_pred = []

    all_uncertainties = []
    concept_uncertainty_sum = np.zeros(test_loader.dataset.num_concepts)
    n_samples = 0

    with torch.no_grad():

        test_bar = tqdm(test_loader, desc="Testing")

        for batch in test_bar:
            images = batch["image"].to(device)
            concepts = batch["concepts"].to(device).float()

            # Forward pass
            outputs = model(images)
            concept_probs = outputs["concept_probs"]
            concept_uncertainty = outputs["concept_uncertainty"]

            # Loss computation
            c_loss = concept_criterion(concept_probs, concepts)

            # Running losses
            test_c_loss += c_loss.item()

            # Predictions
            c_preds = (concept_probs > 0.5)

            test_c_true.extend(concepts.cpu().numpy().flatten())
            test_c_pred.extend(c_preds.cpu().numpy().flatten())

            all_uncertainties.extend(concept_uncertainty.mean(dim=1).cpu().numpy())
            concept_uncertainty_sum += (concept_uncertainty.sum(dim=0).cpu().numpy())
            n_samples += images.size(0)

            # Running accuracies
            test_c_correct += (c_preds == concepts.bool()).sum().item()
            test_c_total += concepts.numel()

            # Progress bar update
            test_bar.set_postfix(
                loss=f"{c_loss.item():.4f}",
                c_acc=f"{100*test_c_correct/max(test_c_total, 1):.2f}%"
            )

    # Final dataset metrics
    test_c_loss_epoch = test_c_loss / len(test_loader)

    test_c_acc = test_c_correct / test_c_total

    test_c_precision = precision_score(test_c_true, test_c_pred, zero_division=0)
    test_c_recall = recall_score(test_c_true, test_c_pred, zero_division=0)
    test_c_f1 = f1_score(test_c_true, test_c_pred, zero_division=0)

    mean_uncertainty = np.mean(all_uncertainties)
    std_uncertainty = np.std(all_uncertainties)
    avg_uncertainty_per_concept = (concept_uncertainty_sum / n_samples)
    top_uncertain = np.argsort(avg_uncertainty_per_concept)[-10:][::-1]

    # Console Output
    print("\n" + "=" * 60)
    print(f"FOLD {fold} TEST RESULTS")
    print("=" * 60)
    print(f"C_Loss: {test_c_loss_epoch:.4f}")
    print(f"C Accuracy : {100*test_c_acc:.2f}%")
    print(f"C Precision: {100*test_c_precision:.2f}%")
    print(f"C Recall   : {100*test_c_recall:.2f}%")
    print(f"C F1 Score : {100*test_c_f1:.2f}%")
    print("=" * 60)
    print(f"\nMean uncertainty : {mean_uncertainty:.4f}")
    print(f"Std uncertainty  : {std_uncertainty:.4f}")

    print("\nTop 10 most uncertain concepts:")

    for idx in top_uncertain:
        print(
            f"Concept {idx}: "
            f"{avg_uncertainty_per_concept[idx]:.6f}"
        )
    
    return {
            "test_c_loss": test_c_loss_epoch,

            "test_c_acc": test_c_acc,
            "test_c_precision": test_c_precision,
            "test_c_recall": test_c_recall,
            "test_c_f1": test_c_f1,

            "mean_uncertainty": mean_uncertainty,
            "std_uncertainty": std_uncertainty,

            "top_uncertain": top_uncertain.tolist(),
            "avg_uncertainty_per_concept": avg_uncertainty_per_concept.tolist(),
        }

def main(batch_size=32, SEED=42, n_folds=5, dataset_name="cub"):
    """
    Evaluate all ProbCBM concept predictor folds and summarize uncertainty.
    """
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = batch_size
    
    RESULTS_DIR = f"results/{dataset_name}/prob_concepts_cv/seed{SEED}"
    SUMMARY_DIR = f"results/{dataset_name}/prob_concepts_cv"

    results = []
    uncertainty_info = []

    if dataset_name.lower() == "cub":
        SPLIT_ROOT = "data/cub/class_attr_data_10"
        IMAGE_ROOT = "data/cub/CUB_200_2011/images"

        # Dataset
        test_ds = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="test", transform=get_transforms("test"))
    
    elif dataset_name.lower() == "awa2":
        ROOT = "data/awa2/Animals_with_Attributes2"

        full_ds = AwA2Dataset(root=ROOT, split="all", transform=None,)
        test_idx = np.load("data/awa2/test_idx.npy")

        test_ds = TransformSubset(full_ds, test_idx, get_transforms("test"),)
    
    elif dataset_name.lower() == "derm7pt":
        ROOT = "data/derm7pt/release_v0"
        test_ds = Derm7ptDataset(root=ROOT, split="test", transform=get_transforms("test"),)    
    
    elif dataset_name.lower() == "celeba":
        ROOT = "data/celeba"
        test_ds = CelebADataset(root=ROOT, split="test", transform=get_transforms("test"),)      
        test_idx = np.load("data/celeba/celeba_20_percent_test_idx.npy")
        test_ds.samples = [test_ds.samples[i] for i in test_idx]

    elif dataset_name.lower() == "synthetic":
            ROOT = "data/synthetic_dataset"
            test_ds = SyntheticDataset(root=ROOT, split="test", transform=get_transforms("test"),)            
    
    print(f"{dataset_name.upper()} TESTING PROB_CONCEPTS_CV MODEL")
    print("Using device: ", DEVICE)
    print("Seed: ", SEED)

    # Dataset and DataLoader
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_ds)}")    

    for fold in range(1, n_folds + 1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        model_path = os.path.join(RESULTS_DIR, f"prob_concept_predictor_fold{fold}_seed{SEED}.pth")
        metrics = evaluate_fold(model_path=model_path, test_loader=test_loader, device=DEVICE, fold=fold)
        metrics["fold"] = fold

        uncertainty_info.append({
            "fold": fold,
            "top_uncertain": metrics.pop("top_uncertain"),
            "avg_uncertainty_per_concept": metrics.pop("avg_uncertainty_per_concept")
        })

        results.append(metrics)
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION TEST SUMMARY")
    print("=" * 80)
    print(f"Models evaluated: {n_folds}")

    best_fold = results_df["test_c_acc"].idxmax()
    print(
        f"Best fold: "
        f"{results_df.loc[best_fold, 'fold']} "
        f"(Acc={100*results_df.loc[best_fold, 'test_c_acc']:.2f}%)"
    )
    
    summary_rows = []

    for col in results_df.columns:
        if col == "fold":
            continue

        mean = results_df[col].mean()
        std = results_df[col].std()

        if ("acc" in col or "precision" in col or "recall" in col or "f1" in col):
            print(
                f"{col}: "
                f"{100*mean:.2f}% ± {100*std:.2f}%"
            )
            mean_std = (
                f"{100*mean:.2f}% ± "
                f"{100*std:.2f}%"
            )
        else:
            print(
                f"{col}: "
                f"{mean:.4f} ± {std:.4f}"
            )
            mean_std = (
                f"{mean:.4f} ± "
                f"{std:.4f}"
            )
        summary_rows.append({
            "metric": col,
            "mean": mean,
            "std": std,
            "mean_std": mean_std
        })
    
    results_df.to_csv(os.path.join(SUMMARY_DIR, f"test_results_prob_concepts_cv_seed{SEED}.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(SUMMARY_DIR, f"test_summary_prob_concepts_cv_seed{SEED}.csv"), index=False)

    uncertainty_df = pd.DataFrame(uncertainty_info)
    uncertainty_df.to_csv(os.path.join(SUMMARY_DIR, f"uncertainty_prob_concepts_cv_seed{SEED}.csv",), index=False,)
    
    summary_txt = os.path.join(SUMMARY_DIR, f"test_results_prob_concepts_cv_seed{SEED}.txt")
    with open(summary_txt, "w") as f:
        f.write("PER-FOLD RESULTS\n\n")

        for _, row in results_df.iterrows():
            f.write(
                f"Fold {int(row['fold'])}: "
                f"C Acc={100*row['test_c_acc']:.2f}% | "
                f"C F1={100*row['test_c_f1']:.2f}%\n"
            )
        f.write("\n")
        f.write("=" * 60)
        f.write("\n\n")
        f.write("MEAN ± STD\n\n")
        for row in summary_rows:
            f.write(
                f"{row['metric']}: "
                f"{row['mean_std']}\n"
                )
        
        f.write("TOP UNCERTAIN CONCEPTS PER FOLD\n\n")

        for fold_info in uncertainty_info:
            f.write(f"Fold {fold_info['fold']}:\n")

            for idx in fold_info["top_uncertain"]:
                unc = fold_info["avg_uncertainty_per_concept"][idx]
                f.write(f"Concept {idx}: {unc:.6f}\n")
            f.write("\n")
    
    print("\nResults saved.")

if __name__ == "__main__":
    main()