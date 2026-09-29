"""
Configuration parameters for MoViNet Human Activity Recognition.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LABELS_PATH = DATA_DIR / "kinetics_600_labels.txt"
SAMPLE_VIDEOS_DIR = BASE_DIR / "sample_videos"

# Official MoViNet Models on Kaggle / TFHub
# Reference: Mobile Video Networks for Efficient Video Recognition (Google Research, CVPR 2021)
MODEL_REGISTRY = {
    "movinet_a0_base": {
        "name": "MoViNet-A0 (Base)",
        "kaggle_handle": "google/movinet/tensorFlow2/a0-base-kinetics-600-classification",
        "tfhub_handle": "https://tfhub.dev/google/movinet/a0/base/kinetics-600/classification/3",
        "resolution": 172,
        "default_frames": 16,
        "description": "Ultra-lightweight and fast, optimized for mobile/CPU real-time inference.",
        "params": "3.1M"
    },
    "movinet_a2_base": {
        "name": "MoViNet-A2 (Base)",
        "kaggle_handle": "google/movinet/tensorFlow2/a2-base-kinetics-600-classification",
        "tfhub_handle": "https://tfhub.dev/google/movinet/a2/base/kinetics-600/classification/3",
        "resolution": 224,
        "default_frames": 16,
        "description": "High accuracy model with 224x224 input resolution for detailed action recognition.",
        "params": "5.3M"
    }
}

DEFAULT_MODEL_KEY = "movinet_a0_base"

# Inference Defaults
DEFAULT_FPS = 5
DEFAULT_NUM_FRAMES = 16
DEFAULT_CONFIDENCE_THRESHOLD = 0.05
DEFAULT_TOP_K = 5

# Human Activity Groupings for Kinetics-600 Filtering & Insights
ACTIVITY_CATEGORIES = {
    "Fitness & Gym": [
        "push up", "pull ups", "jumping jacks", "squat", "lunges", "situp", 
        "deadlifting", "bench pressing", "clean and jerk", "exercising with exercise ball", 
        "bicep curls", "burpees", "plank", "exercising arm", "doing aerobics"
    ],
    "Sports & Athletics": [
        "playing basketball", "playing tennis", "kicking soccer ball", "golf putting",
        "archery", "bowling", "baseball pitch", "badminton", "catching or throwing baseball",
        "catching or throwing frisbee", "dribbling basketball", "shooting basketball",
        "skateboarding", "skiing", "snowboarding", "surfing", "volleyball"
    ],
    "Dance & Performance": [
        "breakdancing", "belly movement", "salsa dancing", "tango dancing",
        "tap dancing", "krumping", "jumpstyle dancing", "robot dancing",
        "country line dancing", "cheerleading", "gymnastics tumbling"
    ],
    "Music & Instruments": [
        "playing guitar", "playing piano", "playing violin", "playing drums",
        "playing saxophone", "playing flute", "playing cello", "playing accordion",
        "playing clarinet", "playing ukulele", "strumming guitar"
    ],
    "Daily Living & Domestic": [
        "brushing teeth", "washing hair", "washing hands", "cooking", "ironing",
        "reading book", "writing", "drinking", "eating burger", "eating cake",
        "making bed", "mopping floor", "vacuuming floor", "cleaning windows"
    ],
    "Locomotion & Outdoor": [
        "walking", "jogging", "running", "riding bike", "climbing ladder",
        "hiking", "swimming", "jumping into pool", "canoeing or kayaking"
    ]
}
