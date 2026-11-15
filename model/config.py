import os
import sys

# ==================== Dynamically detect Dummy mode ====================
USE_DUMMY = "--dummy" in sys.argv

EXPERIMENT_NAME = os.environ.get("EXP_NAME", "lam1e-1_seed42")
SEED = int(os.environ.get("SEED", 42))
PHYSICS_SCALE = float(os.environ.get("PHYSICS_SCALE", 0.1))
USE_PHYSICS = os.environ.get("USE_PHYSICS", "True").lower() == "true"
USE_AUGMENT = os.environ.get("USE_AUGMENT", "True").lower() == "true"

_constraints_str = os.environ.get("ENABLED_CONSTRAINTS", "")
if _constraints_str.strip() == "" or _constraints_str.lower() == "none":
    ENABLED_CONSTRAINTS = None
else:
    ENABLED_CONSTRAINTS = set(_constraints_str.split(","))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if USE_DUMMY:
    DATA_DIR = os.path.join(BASE_DIR, "data", "dummy")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints_dummy", EXPERIMENT_NAME)
else:
    DATA_DIR = os.path.join(BASE_DIR, "data", "split_7d")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints_7d", EXPERIMENT_NAME)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==================== Data parameters ====================
TIME_STEPS = 30
VARIABLES = 8
NUM_INDICES = 6
SPATIAL_SIZE = (80, 140)

PRELOAD = False
NUM_WORKERS = 0
PIN_MEMORY = True

# ==================== Model parameters ====================
D_MODEL = 128
N_HEADS = 4
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 3))
D_FF = 512
DROPOUT = 0.3

# ==================== Training parameters ====================
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-3
GRAD_CLIP = 1.0
EARLY_STOP = 50
WARMUP_EPOCHS = 50
PHYSICS_WARMUP = 30

# ==================== Dummy mode ====================
if USE_DUMMY:
    NUM_EPOCHS = 1
    WARMUP_EPOCHS = 0
    EARLY_STOP = 1
    BATCH_SIZE = 2