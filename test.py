import argparse

from testing.test_joint_cv import main as joint_test_cv
from testing.test_concepts_cv import main as concepts_test_cv
from testing.test_sequential_cv import main as sequential_test_cv
from testing.test_cem_cv import main as cem_main_cv
from testing.test_prob_concepts_cv import main as prob_concepts_cv_main
from testing.test_prob_classifier_cv import main as prob_classifier_cv_main
from testing.test_scbm_cv import main as scbm_cv_main

def main():
    parser = argparse.ArgumentParser(description="Evaluate Concept Bottleneck Models using cross-validation.")

    parser.add_argument("--model", choices=["joint_cv", "concepts_cv", "sequential_cv", "cem_cv", 
                                            "prob_concepts_cv", "prob_classifier_cv", "scbm_cv"], required=True, help="Which model to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--dataset", choices=["cub", "awa2", "derm7pt", "celeba", "synthetic"], default="cub", help="Dataset to evaluate.")
    args = parser.parse_args()

    TESTERS = {
        "joint_cv": joint_test_cv,
        "concepts_cv": concepts_test_cv,
        "sequential_cv": sequential_test_cv,
        "cem_cv": cem_main_cv,
        "prob_concepts_cv": prob_concepts_cv_main,
        "prob_classifier_cv": prob_classifier_cv_main,
        "scbm_cv": scbm_cv_main,
    }

    tester = TESTERS[args.model]

    tester(
        batch_size=args.batch_size,
        SEED=args.seed,
        n_folds=args.folds,
        dataset_name=args.dataset,
    )
if __name__ == "__main__":
    main()