import os
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from training.joint_training_cv import main as joint_main_cv
from training.train_concepts_cv import main as concepts_main_cv
from training.train_sequential_cv import main as sequential_main_cv
from training.cbm_cross_validation import run_cv
from training.cem_training_cv import main as cem_main_cv
from training.train_prob_concepts_cv import main as prob_concepts_main_cv
from training.train_prob_classifier_cv import main as prob_classifier_main_cv
from training.scbm_training_cv import main as scbm_main_cv

def main():
    parser = argparse.ArgumentParser(description="Train Concept Bottleneck Models using cross-validation.")

    parser.add_argument("--model", choices=["joint_cv", "concepts_cv", "sequential_cv", "cem_cv", 
                                            "prob_concepts_cv", "prob_classifier_cv", "scbm_cv"], required=True, help="Model to train.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--dataset", choices=["cub", "awa2", "derm7pt", "celeba", "synthetic"], default="cub", help="Dataset to use.")
    parser.add_argument("--start_fold", type=int, default=1, help="Fold to start cross-validation from.")

    args = parser.parse_args()

    TRAINERS = {
        "joint_cv": joint_main_cv,
        "concepts_cv": concepts_main_cv,
        "sequential_cv": sequential_main_cv,
        "cem_cv": cem_main_cv,
        "prob_concepts_cv": prob_concepts_main_cv,
        "prob_classifier_cv": prob_classifier_main_cv,
        "scbm_cv": scbm_main_cv,
    }

    trainer = TRAINERS[args.model]

    run_cv(
        trainer=trainer,
        experiment_name=f"{args.dataset}/{args.model}",
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        SEED=args.seed,
        n_splits=args.folds,
        start_fold=args.start_fold,
    )

if __name__ == "__main__":
    main()