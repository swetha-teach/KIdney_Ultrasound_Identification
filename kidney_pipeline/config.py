from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PATIENTS_DIR = ROOT_DIR / "patient_folders"
REFERENCE_DIR = ROOT_DIR / "reference_kidney"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

COSINE_K_VALUES = [5, 10, 20]
SVM_NU_VALUES = [0.01, 0.05, 0.10, 0.20]
GAP_MIN_THRESHOLD = 0.03

GROUND_TRUTH = {
    "patient001": [3, 4],
    "patient002": [4, 5, 6],
    "patient003": [5, 6],
    "patient004": [1, 9],
    "patient005": [3, 4, 5],
    "patient006": [4, 5, 6, 7],
    "patient007": [4, 5, 6],
    "patient008": [8, 9],
    "patient009": [4, 6],
    "patient010": [4, 6],
    "patient011": [4, 6],
    "patient012": [2, 3, 4, 5, 6],
    "patient013": [4, 6],
    "patient014": [11, 12, 13, 14, 15],
    "patient015": [1, 2, 3, 4, 5, 6],
    "patient016": [2, 3],
    "patient017": [2, 4],
    "patient018": [2, 3],
    "patient019": [2, 4],
    "patient020": [2, 3, 4, 5],
}

EXTRACTOR_NAMES = [
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "convnext_tiny",
    "dinov2_vits14",
    "ultrasam",
]


def get_image_paths(folder: Path) -> list:
    return sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ])


def frame_number(image_path: Path) -> int:
    return int(image_path.stem.split("-")[-1])
