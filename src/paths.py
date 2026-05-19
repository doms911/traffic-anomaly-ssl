"""Global filesystem paths. Edit once per machine."""
from pathlib import Path

# Root paths
PROJECT_ROOT = Path("/lustre/home/dbarukci/projects/traffic-anomaly-ssl")
DATA_ROOT = Path("/lustre/home/dbarukci/datasets")

# Dataset paths
CITYSCAPES_ROOT = DATA_ROOT / "cityscapes"
CITYSCAPES_SEQUENCE_ROOT = DATA_ROOT / "cityscapes"  # ista lokacija, drugi pod-folder
FISHYSCAPES_ROOT = DATA_ROOT / "fishyscapes"

# Output paths
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
LOGS_ROOT = PROJECT_ROOT / "logs"

# Auto-create output dirs
CHECKPOINTS_ROOT.mkdir(exist_ok=True)
LOGS_ROOT.mkdir(exist_ok=True)