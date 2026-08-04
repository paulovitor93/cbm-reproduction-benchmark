from generator import RuleBasedSceneGenerator
from extractor import ConceptExtractor
from renderer_png import SceneRenderer
from dataset_builder import DatasetBuilder
from split import DatasetSplitter
from pathlib import Path

# Configuration
GENERATION_SEED = 3938019914

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_dataset"

SAMPLES_PER_CLASS = 1000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create components
generator = RuleBasedSceneGenerator()
renderer = SceneRenderer()
extractor = ConceptExtractor()

# Generate dataset
builder = DatasetBuilder(generator=generator, renderer=renderer, extractor=extractor,)
builder.build(output_dir=OUTPUT_DIR, samples_per_class=SAMPLES_PER_CLASS, generation_seed=GENERATION_SEED,)

# Create official split
splitter = DatasetSplitter()

splitter.split(
    metadata_csv=OUTPUT_DIR / "metadata" / "metadata.csv",
    output_dir=OUTPUT_DIR,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
)

print("\nDataset generation complete!")