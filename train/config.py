import torch
import os
from pathlib import Path

os.environ["HF_HOME"] = "/project2/jieyuz_1727/snehansh/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/project2/jieyuz_1727/snehansh/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "/project2/jieyuz_1727/snehansh/hf_transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/project2/jieyuz_1727/snehansh/hf_hub"

PROJECT_ROOT = Path("/project2/jieyuz_1727/snehansh")
OUTPUT_DIR = PROJECT_ROOT / "outputs_train_new"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "resnet50"
NUM_CLASSES = 1000

EPOCHS = 100
BATCH_SIZE_PER_GPU = 640
NUM_WORKERS = 6

OPTIMIZER = "sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NESTEROV = True

MAX_LR = 0.18
SCALE_LR_WITH_GPUS = False 

ONECYCLE_PCT_START = 0.30
ONECYCLE_DIV_FACTOR = 25.0
ONECYCLE_FINAL_DIV_FACTOR = 1e4

IMG_SIZE = 224
RESIZE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

USE_RANDOM_RESIZED_CROP = True
USE_HORIZONTAL_FLIP = True

USE_COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.4
COLOR_JITTER_CONTRAST = 0.4
COLOR_JITTER_SATURATION = 0.4
COLOR_JITTER_HUE = 0.1

USE_AUTOAUGMENT = True

USE_RANDOM_ERASING = True
RANDOM_ERASING_PROB = 0.25

USE_MIXUP = True
MIXUP_ALPHA = 0.2
USE_CUTMIX = True
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

LABEL_SMOOTHING = 0.1

WORLD_SIZE = torch.cuda.device_count()

USE_AMP = True

SAVE_FREQ = 5  
LOG_FREQ = 100  
RESUME = True  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

def get_scaled_lr():
    if SCALE_LR_WITH_GPUS:
        return MAX_LR * WORLD_SIZE
    return MAX_LR

def get_total_batch_size():
    return BATCH_SIZE_PER_GPU * WORLD_SIZE
