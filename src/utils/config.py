# src/utils/config.py

import os

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Data
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "src", "model", "intent_classifier")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Inference
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

# Misc
SEED = 42
