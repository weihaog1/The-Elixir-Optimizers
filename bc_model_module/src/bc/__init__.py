"""BC model package - behavior cloning for Clash Royale card placement."""

from src.bc.feature_extractor import CRFeatureExtractor
from src.bc.bc_policy import BCPolicy
from src.bc.bc_dataset import BCDataset, load_datasets
from src.bc.train_bc import BCTrainer, TrainConfig

# Live inference (optional - may fail if mss/pyautogui not installed)
try:
    from src.bc.live_inference import LiveConfig, LiveInferenceEngine
except ImportError:
    LiveConfig = None  # type: ignore[assignment, misc]
    LiveInferenceEngine = None  # type: ignore[assignment, misc]

__all__ = [
    "CRFeatureExtractor",
    "BCPolicy",
    "BCDataset",
    "load_datasets",
    "BCTrainer",
    "TrainConfig",
    "LiveConfig",
    "LiveInferenceEngine",
]
