import argparse
import subprocess
import os
import time
import sys

ALL_DATASETS = ["cub", "awa2", "derm7pt", "celeba", "synthetic",]

ALL_MODELS = ["joint_cv", "concepts_cv", "sequential_cv", "cem_cv", 
              "prob_concepts_cv", "prob_classifier_cv", "scbm_cv",]

DEFAULT_SEEDS = [203040, 532164]

def main():
    pipeline_start = time.time()
    parser = argparse.ArgumentParser(description="Run the complete CBM benchmark pipeline.")

    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all"] + ALL_DATASETS, help="Datasets to benchmark.",)

    parser.add_argument("--models", nargs="+", default=["all"], choices=["all"] + ALL_MODELS, help="Models to execute.",)

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")

    parser.add_argument("--folds", type=int, default=5,)

    parser.add_argument("--batch_size", type=int, default=32,)

    parser.add_argument("--seeds", nargs="+", type=int, default=[203040, 532164],)

    parser.add_argument("--start_fold", type=int, default=1, help="Resume cross-validation from this fold.")

    parser.add_argument("--run_all", action="store_true", help="Run every dataset, model and default seed.")

    args = parser.parse_args()

    if args.run_all:
        datasets = ALL_DATASETS
        models = ALL_MODELS
        seeds = DEFAULT_SEEDS

    else:
        datasets = (ALL_DATASETS if "all" in args.datasets else args.datasets)
        models = (ALL_MODELS if "all" in args.models else args.models)

    folds = args.folds
    epochs = args.epochs
    batch_size = args.batch_size

    print("=" * 80)
    print("CBM BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Datasets   : {datasets}")
    print(f"Models     : {models}")
    print(f"Epochs     : {epochs}")
    print(f"Batch size : {batch_size}")
    print(f"Folds      : {folds}")
    print(f"Seeds      : {seeds}")
    print(f"Start fold : {args.start_fold}")
    print("=" * 80)

    total_trainings = (len(datasets) * len(models) * len(seeds) * folds)
    print(f"Total trainings: {total_trainings}")

    for dataset in datasets:

        # save seeds
        SEED_DIR = f"results/{dataset}"
        os.makedirs(SEED_DIR, exist_ok=True)

        with open(os.path.join(SEED_DIR, "random_seeds.txt"), "w") as f:
            for s in seeds:
                f.write(f"{s}\n")

        with open(os.path.join(SEED_DIR, "benchmark_config.txt"), "w") as f:
            f.write(f"Datasets   : {datasets}\n")
            f.write(f"Models     : {models}\n")
            f.write(f"Epochs     : {epochs}\n")
            f.write(f"Batch size : {batch_size}\n")
            f.write(f"Folds      : {folds}\n")
            f.write(f"Seeds      : {seeds}\n")
            f.write(f"Start fold : {args.start_fold}\n")

        # run experiments
        for seed in seeds:

            print("\n" + "=" * 80)
            print(f"DATASET = {dataset.upper()}")
            print(f"SEED    = {seed}")
            print("=" * 80)

            for i, model in enumerate(models, start=1):
                model_start = time.time()

                print("\n" + "=" * 80)
                print(
                    f"\n[{i}/{len(models)}] "
                    f"TRAINING {model.upper()} ({dataset.upper()})"
                )
                print("=" * 80)

                subprocess.run([
                    sys.executable,
                    "train.py",
                    "--model", model,
                    "--dataset", dataset,
                    "--epochs", str(epochs),
                    "--batch_size", str(batch_size),
                    "--folds", str(folds),
                    "--seed", str(seed),
                    "--start_fold", str(args.start_fold),
                ], check=True)

                print("\n" + "=" * 80)
                print(f"TESTING {model.upper()} ({dataset.upper()})")
                print("=" * 80)

                subprocess.run([
                    sys.executable,
                    "test.py",
                    "--model", model,
                    "--dataset", dataset,
                    "--batch_size", str(batch_size),
                    "--folds", str(folds),
                    "--seed", str(seed),
                ], check=True)

                elapsed = time.time() - model_start

                print("\n" + "-" * 80)
                print(f"{model.upper()} completed in {elapsed/60:.2f} minutes.")
                print("-" * 80)

    total_time = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS FINISHED")
    print(f"Total execution time: {total_time/3600:.2f} hours")
    print("=" * 80)

if __name__ == "__main__":
    main()