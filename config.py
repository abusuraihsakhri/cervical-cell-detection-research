"""
Central configuration for the pap-smear-cyto-algo project.

Per AGENT-SPEC-FINAL.md Section 2: hardware-driven model choice for a
single 6GB VRAM GPU. Per Section 4: class mapping. Per Section 5:
epoch strategy is a stopping rule, not a fixed target.

Nothing in this file has been benchmarked on the actual target GPU yet.
Per Section 5, log real per-epoch wall-clock time from the first few
epochs and treat any timing assumption here as a guess until then.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path & Storage Helpers (Defensive I/O against unmounted/locked drives)
# ---------------------------------------------------------------------------
def resolve_storage_path(target_path_str: str) -> Path:
    """Safely convert a storage path string to Path without crashing on offline media."""
    try:
        return Path(target_path_str)
    except OSError as err:
        logger.warning("Storage path '%s' cannot be resolved: %s", target_path_str, err)
        return Path(target_path_str)


def is_path_accessible(path: Path | None) -> bool:
    """Safely check whether a filesystem path exists and is accessible."""
    if path is None:
        return False
    try:
        return path.exists()
    except OSError as err:
        logger.warning("Path '%s' is inaccessible or drive is locked: %s", path, err)
        return False


# ---------------------------------------------------------------------------
# Secure API Key Management
# ---------------------------------------------------------------------------
def get_roboflow_key() -> str:
    """Retrieve Roboflow API key from OS environment, Windows User Registry, or .env with validation."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if (not key or key in ("your_key_here", "your_roboflow_api_key_here")) and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                val, _ = winreg.QueryValueEx(k, "ROBOFLOW_API_KEY")
                if val:
                    key = str(val).strip()
        except OSError:
            pass

    if not key or key in ("your_key_here", "your_roboflow_api_key_here"):
        raise EnvironmentError(
            "ROBOFLOW_API_KEY is not configured or is set to a placeholder. "
            "Please set the ROBOFLOW_API_KEY environment variable securely."
        )
    return key


# ---------------------------------------------------------------------------
# Model Weights Cryptographic Integrity Verification (OWASP A08:2025)
# ---------------------------------------------------------------------------
KNOWN_WEIGHTS_SHA256 = {
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "models/best.pt": "9f6245bf9d80870155c7edcb4292e3f87deb989e30e7fc0499ebf17aac9597e1",
    "expc1_sipakmed_2class/weights/best.pt": "4a68db84841262b6299b1ef5b0b5dc412b69616c5adc0c21aa08c07c2adcda97",
    "expc2b_combined_150/weights/best.pt": "1d3e0ca25376b9a25b41a93511f063ec8ea22688feff523cdd4569703e59669e",
}


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_model_checksum(weights_path: Path | str, expected_sha256: str | None = None, enforce: bool = False) -> bool:
    """
    Verify the cryptographic SHA-256 integrity of model weights before loading.
    
    If expected_sha256 is not explicitly passed, looks up known weights by normalized path ending.
    If enforce=True, raises ValueError on mismatch. Otherwise logs a warning.
    """
    p = Path(weights_path)
    if not is_path_accessible(p):
        return False

    expected = expected_sha256
    if not expected:
        norm_str = str(p.resolve()).replace("\\", "/").lower()
        for k, v in KNOWN_WEIGHTS_SHA256.items():
            if norm_str.endswith(k.lower()):
                expected = v
                break
    if not expected:
        # Custom checkpoint without a pre-registered hash
        return True

    actual = compute_file_sha256(p)
    if actual != expected.lower():
        msg = f"Integrity check FAILED for {p.name}: expected {expected}, got {actual}"
        if enforce:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    logger.info("Integrity check PASSED for %s (%s)", p.name, actual[:12])
    return True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SIPAKMED_DIR = DATA_DIR / "sipakmed"
APCDATA_DIR = DATA_DIR / "apcdata"
# HMCHH-TCT-CellDet is too large for the project's C: drive checkout;
# prepare_hmchh.py was run against the raw files staged on D: and wrote its
# YOLO-format output there too (see run_prepare_hmchh.bat) — not a placeholder.
HMCHH_TCT_DIR = resolve_storage_path("D:/pap_model/HMCHH_YOLO_prepared")
CRIC_DIR = DATA_DIR / "cric"
MENDELEY_LBC_DIR = DATA_DIR / "mendeley_lbc"
HERLEV_DIR = DATA_DIR / "herlev"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

# Real data.yaml locations produced by the prepare_*.py scripts — used as
# argparse defaults in evaluation/cross_dataset_eval.py so it can be run
# without repeating these paths on the command line.
SIPAKMED_DATA_YAML = SIPAKMED_DIR / "mirror_a" / "data.yaml"
APCDATA_DATA_YAML = APCDATA_DIR / "APCData_YOLO_prepared" / "data.yaml"
HMCHH_DATA_YAML = HMCHH_TCT_DIR / "data.yaml"

# ---------------------------------------------------------------------------
# Phase 2 second-dataset resolution — Section 3 & Section 9 open item
# ---------------------------------------------------------------------------
# Resolved via web research (Aug 2026), replacing the original CRIC-first /
# Mendeley-LBC-fallback plan with a decision actually grounded in each
# candidate's real annotation format:
#
#   - CRIC (database.cric.com.br, figshare DOI 10.6084/m9.figshare.c.4960286.v2):
#     CONFIRMED point-click cell-center coordinates, NOT bounding boxes.
#     Cannot be used for YOLO detection without converting points into
#     synthetic fixed-radius boxes — a real methodological choice with its
#     own bias (box size becomes a hyperparameter you're inventing), not
#     "free" bbox data. This resolves spec Section 9's open CRIC-verification
#     item: verified, and the answer is "not directly usable."
#   - Mendeley LBC (data.mendeley.com/datasets/zddtpgzv63, the dataset the
#     spec's literature review actually meant): confirmed used across
#     multiple published papers as a CLASSIFICATION dataset (folder-per-class
#     images), no bounding-box annotations. Not usable for detection as-is.
#   - APCData (data.mendeley.com/datasets/ytd568rh3p, DOI 10.17632/ytd568rh3p.1):
#     genuine YOLO-format .txt bounding boxes, ready to use with zero
#     conversion. 425 images, ~3,619 annotated cells, 6-class Bethesda
#     system (NILM/ASC-US/ASC-H/LSIL/HSIL/SCC). SELECTED as the Phase 2
#     primary target — it's the only public candidate found with real
#     detection-ready annotations.
#   - HMCHH-TCT-CellDet (figshare DOI 10.6084/m9.figshare.27901206):
#     genuine XML bounding boxes, 8,037 images, ~15,761 boxes, but BINARY
#     classes only (abnormal/normal). Kept as an optional third source —
#     good volume for a coarse 2-class cross-check via ROLLUP_3_TO_2 below,
#     not for 5-class comparison.
PHASE2_DATASET = "apcdata"  # was: "cric" / "mendeley_lbc" placeholder
PHASE2_DATASET_SOURCES = {
    "apcdata": {
        "name": "APCData: Cervical cytology cells",
        "url": "https://data.mendeley.com/datasets/ytd568rh3p/1",
        "doi": "10.17632/ytd568rh3p.1",
        "annotation_format": "YOLO .txt bounding boxes (APCData_YOLO folder) — ready to use",
        "images": 425,
        "annotated_cells": 3619,
        "classes_bethesda_6": ["NILM", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC"],
    },
    "hmchh_tct_celldet": {
        "name": "HMCHH-TCT-CellDet",
        "url": "https://doi.org/10.6084/m9.figshare.27901206",
        "annotation_format": "XML bounding boxes — needs conversion to YOLO .txt",
        "images": 8037,
        "annotated_boxes": 15761,
        "classes_binary": ["Normal", "Abnormal"],
    },
    "cric": {
        "name": "CRIC Searchable Image Database",
        "url": "https://database.cric.com.br",
        "doi": "10.6084/m9.figshare.c.4960286.v2",
        "annotation_format": "Point-click cell-center (x, y) — NOT bounding boxes; needs synthetic-box conversion",
        "status": "verified unusable as-is for YOLO detection (Section 9 resolved)",
    },
    "mendeley_lbc": {
        "name": "Liquid based-cytology Pap smear dataset",
        "url": "https://data.mendeley.com/datasets/zddtpgzv63/2",
        "doi": "10.17632/zddtpgzv63.2",
        "annotation_format": "Classification folders (image-level label only) — NOT bounding boxes",
        "status": "confirmed unusable for detection as-is",
    },
}

# ---------------------------------------------------------------------------
# Roboflow (SIPaKMeD acquisition — Section 3)
# ---------------------------------------------------------------------------
# Two mirrors to diff and pick the more complete one as primary.
ROBOFLOW_SIPAKMED_MIRRORS = {
    "mirror_a": {"workspace": "ik-zu-quan-o9tdm", "project": "sipakmed-ioflq", "version": 1},
    "mirror_b": {"workspace": "sipakmed", "project": "sipakmed-t9emb", "version": 1},
}

# ---------------------------------------------------------------------------
# Model choice — Section 2
# ---------------------------------------------------------------------------
# YOLOv8n is primary. Escalate to YOLOv8s ONLY if Phase 1 baseline proves
# insufficient. Do not skip straight to larger models. Nothing above
# YOLOv8s fits the 6GB VRAM budget per the spec's own table.
PRIMARY_MODEL = "yolov8n.pt"
FALLBACK_MODEL = "yolov8s.pt"  # escalate only on documented Phase 1 shortfall
MAX_ALLOWED_MODEL = "yolov8s.pt"  # hard ceiling — no larger model without an explicit decision

# ---------------------------------------------------------------------------
# Mandatory training flags — Section 2
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "half": True,          # FP16, ~40-50% memory reduction
    "batch": 16,            # drop to 8, then 4, if OOM
    "imgsz": 640,           # do not raise to 1280 during training on this card
    "cache": "disk",        # avoid full in-memory cache as dataset grows
    "workers": 4,
    "epochs": 200,           # ceiling / safety cap only — see Section 5
    "patience": 30,          # stop if val mAP@50 hasn't improved in 30 epochs
}

# Batch sizes to try in order on OOM (Section 2: "drop to 8, then 4, if OOM")
BATCH_FALLBACK_SEQUENCE = [16, 8, 4]

# ---------------------------------------------------------------------------
# Class mapping — Section 4
# ---------------------------------------------------------------------------
# Primary target: 5-class native SIPaKMeD taxonomy. Matches most published
# benchmarks for direct comparison. 3-class and 2-class roll-ups are
# derived post-hoc from the SAME trained model's confusion matrix —
# never train separate models for them.
CLASSES_5 = [
    "Superficial-Intermediate",
    "Parabasal",
    "Koilocytotic",
    "Dyskeratotic",
    "Metaplastic",
]

# Roll-up: 5-class -> 3-class (Normal / Benign / Abnormal)
# NOTE: this mapping is a first-pass clinical grouping and should be
# reviewed against SIPaKMeD's own documentation / Ares's judgment before
# being treated as authoritative. Superficial-Intermediate and Parabasal
# are normal squamous maturation stages; Koilocytotic and Dyskeratotic
# are the classically abnormal/precancerous categories; Metaplastic is
# a benign reactive process, not squamous-normal and not abnormal.
ROLLUP_5_TO_3 = {
    "Superficial-Intermediate": "Normal",
    "Parabasal": "Normal",
    "Metaplastic": "Benign",
    "Koilocytotic": "Abnormal",
    "Dyskeratotic": "Abnormal",
}

# Roll-up: 3-class -> 2-class (Normal / Abnormal). Benign folds into
# Normal here since it is not a squamous abnormality; revisit if Ares
# wants Benign folded into Abnormal instead (more conservative screen).
ROLLUP_3_TO_2 = {
    "Normal": "Normal",
    "Benign": "Normal",
    "Abnormal": "Abnormal",
}

CLASSES_3 = ["Normal", "Benign", "Abnormal"]
CLASSES_2 = ["Normal", "Abnormal"]

# Clinically significant confusion pair to always call out explicitly
# (Section 6a) rather than burying in aggregate accuracy.
WATCH_CONFUSION_PAIRS = [
    ("Dyskeratotic", "Koilocytotic"),
    ("Dyskeratotic", "Metaplastic"),
]

# ---------------------------------------------------------------------------
# Default inference confidence threshold.
#
# Detection RECALL, not precision, is this model's binding constraint (see
# results/Phase1-3_Training_Evaluation_Report.docx Section 5.1/5.4): precision
# is fairly flat (0.40-0.56) across all 5 SIPaKMeD classes, while recall is
# what varies and drives overall quality. Section 6d's cost-weighted PR-curve
# analysis on the clinically significant Dyskeratotic class found 0.111 as the
# threshold minimizing (10 * false_negatives + false_positives), vs. the prior
# arbitrary default of 0.25 — raising Dyskeratotic recall from ~0.51 to 0.624
# at the cost of more false positives. That N=10 cost ratio is itself an
# unvalidated placeholder (see pr_curve_threshold.py), and only Dyskeratotic's
# curve was analyzed — this is applied as a single global default across all
# classes, not a per-class-tuned threshold.
DEFAULT_INFERENCE_CONF = 0.111

# ---------------------------------------------------------------------------
# Cross-dataset class alignment — APCData (6-class Bethesda) vs. SIPaKMeD
# (5-class native, non-Bethesda) don't share a taxonomy, so zero-shot
# cross_dataset_eval.py can only compare them meaningfully at the 2-class
# (Normal/Abnormal) roll-up level, not at native class granularity. Both
# roll-ups below are FIRST-PASS mappings pending Ares's clinical review —
# not treated as authoritative without confirmation, same caveat as
# ROLLUP_5_TO_3 above.
APCDATA_CLASSES_BETHESDA_6 = ["NILM", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC"]

# NILM (negative for intraepithelial lesion/malignancy) is the only
# Bethesda-normal category here; everything else (atypical squamous cells
# through invasive carcinoma) rolls up to Abnormal for a 2-class comparison.
ROLLUP_APCDATA_6_TO_2 = {
    "NILM": "Normal",
    "ASC-US": "Abnormal",
    "ASC-H": "Abnormal",
    "LSIL": "Abnormal",
    "HSIL": "Abnormal",
    "SCC": "Abnormal",
}

# ---------------------------------------------------------------------------
# External benchmark to always cite when reporting cross-dataset numbers
# (Section 0, Section 6b)
# ---------------------------------------------------------------------------
COSKUN_2026_BENCHMARK = {
    "citation": "Coskun et al., Bioengineering, 2026",
    "accuracy": 0.91,
    "macro_f1": 0.91,
    "note": (
        "Combined SIPaKMeD + Herlev + CRIC + 416 proprietary multi-center "
        "WSIs; ResNet50 classification (not detection); used proprietary "
        "data this project does not have access to. A lower number here "
        "is the expected public-data-only, YOLO-detection ceiling, not a "
        "failure — see Section 0 of the spec."
    ),
}

FUZZY_ENSEMBLE_2026_ECE_BENCHMARK = {
    "citation": "Fuzzy ensemble framework, 2026 (see Section 6c)",
    "ece": 0.030,
    "note": "Classification-based approach; comparison point for this project's detection-based ECE.",
}

# ---------------------------------------------------------------------------
# Banned language (Section 0 / Section 8) — used by report-generation
# scripts to sanity-check output text before it's written.
# ---------------------------------------------------------------------------
BANNED_PHRASES = [
    "clinical grade",
    "clinical-grade",
    "who-compliant",
    "who compliant",
    "diagnostic tool",
    "closes a research gap",
    "closes an unaddressed research gap",
]
