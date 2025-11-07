import torch
import os
from pathlib import Path

os.environ["HF_HOME"] = "/project2/jieyuz_1727/snehansh/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/project2/jieyuz_1727/snehansh/hf_datasets"

PROJECT_ROOT = Path("/project2/jieyuz_1727/snehansh")
OUTPUT_DIR = PROJECT_ROOT / "outputs_finetune_multi"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "resnet50"
NUM_CLASSES = 1000
PRETRAINED_WEIGHTS = "/project2/jieyuz_1727/snehansh/outputs/checkpoints/pretrained.pth"
LOAD_PRETRAINED = True

EPOCHS = 50
BATCH_SIZE_PER_GPU = 640
ACCUMULATION_STEPS = 2  # in-case multi GPUs not available (fall-back)

NUM_WORKERS = 12

OPTIMIZER = "sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NESTEROV = True

LR = 0.002
USE_SCHEDULER = True
SCHEDULER_TYPE = "constant_decay"
DECAY_MIN_LR = 1e-5

IMG_SIZE = 224
RESIZE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

USE_RANDOM_RESIZED_CROP = True
USE_HORIZONTAL_FLIP = True

USE_COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.05

USE_AUTOAUGMENT = True
USE_RANDOM_ERASING = True
RANDOM_ERASING_PROB = 0.25

USE_MIXUP = False
MIXUP_ALPHA = 0.0
USE_CUTMIX = False
CUTMIX_ALPHA = 0.0
CUTMIX_PROB = 0.0
LABEL_SMOOTHING = 0.04

USE_AMP = True
SAVE_FREQ = 2
LOG_FREQ = 50
RESUME = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True

def get_total_batch_size():
    """Effective batch size"""
    return BATCH_SIZE_PER_GPU * ACCUMULATION_STEPS

def print_config():
    print("\n" + "="*80)
    print("SINGLE GPU FINE-TUNING (GRADIENT ACCUMULATION)")
    print("="*80)
    print(f"Pretrained: {PRETRAINED_WEIGHTS}")
    print(f"Device: {DEVICE}")
    print(f"\nPhysical batch: {BATCH_SIZE_PER_GPU}")
    print(f"Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"Effective batch: {get_total_batch_size()}")
    print(f"Epochs: {EPOCHS}")
    print("="*80 + "\n")
