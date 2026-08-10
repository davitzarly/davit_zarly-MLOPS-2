"""Configuration for the NEU-DET Steel Surface Defect Classification TFX pipeline.

This module centralises every constant that the pipeline and the local runner
need so that the pipeline definition itself stays clean and declarative
(clean-code principle: configuration should be separated from logic).
"""
import os

# --------------------------------------------------------------------------- #
# 1. Project identification
# --------------------------------------------------------------------------- #
PIPELINE_NAME = "davit_zarly_pipeline"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# 2. Data location
# --------------------------------------------------------------------------- #
DATA_ROOT = os.path.join(PROJECT_ROOT, "davit_zarly-pipeline", "data")
DATA_FILE = os.path.join(DATA_ROOT, "data.csv")
EVAL_FILE = os.path.join(DATA_ROOT, "eval.csv")

# Base directory that contains the NEU-DET dataset (used by Transform to
# resolve the relative image paths stored in the CSV manifest, e.g.
# "NEU-DET/train/images/crazing/crazing_1.jpg"). The NEU-DET/ folder lives
# directly under the submission root, so the base dir IS the project root.
DATASET_BASE_DIR = os.environ.get("NEU_DET_BASE_DIR", PROJECT_ROOT)

# --------------------------------------------------------------------------- #
# 3. Pipeline artefact locations
# --------------------------------------------------------------------------- #
PIPELINE_ROOT = os.path.join(PROJECT_ROOT, "pipelines", PIPELINE_NAME)
METADATA_PATH = os.path.join(PROJECT_ROOT, "metadata", PIPELINE_NAME, "metadata.db")
SERVING_MODEL_DIR = os.path.join(PROJECT_ROOT, "serving_model", PIPELINE_NAME)

# --------------------------------------------------------------------------- #
# 4. Training hyper-parameters
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 200          # NEU-DET images are 200x200 RGB
NUM_CHANNELS = 3
NUM_CLASSES = 6
BATCH_SIZE = 32
TRAIN_EPOCHS = 20
LEARNING_RATE = 1e-3

LABEL_KEYS = ["label"]
VOCAB = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# --------------------------------------------------------------------------- #
# 5. Beam runner configuration (Apache Beam local runner)
# --------------------------------------------------------------------------- #
BEAM_PIPELINE_ARGS = [
    "--runner=DirectRunner",          # Apache Beam local direct runner
    "--experiments=use_runner_v2",
    "--direct_num_workers=4",
]

# --------------------------------------------------------------------------- #
# 6. Tuner configuration (bonus)
# --------------------------------------------------------------------------- #
TUNER_NUM_TRIALS = 5
TUNER_EPOCHS_PER_TRIAL = 3
