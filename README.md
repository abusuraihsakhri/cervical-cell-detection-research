# Cervical Cell Detection Research: Retrospective Evaluation & Reproducibility

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Research%20Site-14685e?style=flat&logo=github)](https://abusuraihsakhri.github.io/cervical-cell-detection-research/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Release Verification](https://img.shields.io/badge/Verification-29%2F29%20PASS-success)](publication/reproducibility/release_verification.json)
[![Weights Integrity](https://img.shields.io/badge/Model%20Weights-SHA--256%20Certified-blue)](publication/weights/WEIGHTS_MANIFEST.json)

> **Research Use Only:** This repository contains retrospective scientific evaluation code and trained models. These models are not medical devices and have not been validated for screening, diagnosis, or patient-level clinical decision-making.

---

## 🌐 Live Research Site (GitHub Pages)

An interactive, editorial presentation of this research project, including interactive figure viewers, full metrics tables, and direct checkpoint download cards, is hosted on GitHub Pages:

👉 **[https://abusuraihsakhri.github.io/cervical-cell-detection-research/](https://abusuraihsakhri.github.io/cervical-cell-detection-research/)** (source in `docs/`)

---

## 🎯 Project Overview & Core Question

This project investigates how **microscope field overlap**, **incomplete cell annotations (sparse vs. dense)**, and **cross-dataset taxonomy conversions** affect the measured performance of lightweight object detectors (YOLOv8n).

### Key Empirical Findings

1. **Microscope Field Overlap Leakage:** Image comparison and normalized cross-correlation (NCC) identified **791 confirmed overlapping microscope field pairs**. Naive filename-based splitting allowed 10 of 20 evaluation fields to cross partition boundaries. Connected component grouping is required to guarantee strict partition independence.
2. **Annotation Completeness Inversion:** For identical detections on 20 held-out evaluation fields, abnormal-cell precision was **52.4%–87.1%** evaluated against the 1,067-cell human-verified dense reference, but dropped to **20.4%–45.7%** against native sparse labels because valid, unannotated cells were erroneously penalized as false positives.
3. **Cross-Center Domain Degradation:** External evaluation on **1,077 HMCHH fields** (1,837 abnormal cells) revealed abnormal-cell recall between **20.8% and 51.8%**, with reference-matched abnormal precision of only **3.8%–6.7%**, demonstrating significant domain shift across preparation and staining protocols.

---

## 🛡️ Model Checkpoints & Weights Safeguard

To ensure that trained model weights (`.pt`) are **never lost, omitted, or corrupted**, all 6 primary research checkpoints and baselines are permanently tracked in this repository, cataloged in `publication/weights/WEIGHTS_MANIFEST.json`, and un-ignored in `.gitignore`.

### Certified Checkpoints Manifest

| Model Checkpoint | Training Data | Seed | Best Epoch | Size | SHA-256 Checksum |
|---|---|---|---|---|---|
| `publication/weights/v3_combined_seed17/weights/best.pt` | SIPaKMeD + APCData | 17 | 55 | 5.95 MB | `bf1395e4d133e9b21fecfb935ad25e7f902a8f1cc7052feee4442ba42d0e8795` |
| `publication/weights/v3_combined_seed43/weights/best.pt` | SIPaKMeD + APCData | 43 | 68 | 5.95 MB | `474532ea49ec30c6f23d6a91e6b1cee5132935c151cf8ad43e2d27c2804daf05` |
| `publication/weights/v3_combined_seed101/weights/best.pt` | SIPaKMeD + APCData | 101 | 60 | 5.95 MB | `0ce73504628d9c7c72e4110bc5f0159ffc4597a74a93b12c5f29fc5a54cb1b13` |
| `publication/weights/v3_sipakmed_only_seed17/weights/best.pt` | SIPaKMeD Only | 17 | 57 | 5.95 MB | `ce17ee57eaed46ec057eb972c0e1be1cda41653354e62f10b0ba443a0e854a73` |
| `publication/weights/v3_sipakmed_only_seed43/weights/best.pt` | SIPaKMeD Only | 43 | 57 | 5.95 MB | `7b1dd55ba492c7fd28b4132dd5c5c4c766af5fc699d9a3fff09b37fa8be5f17b` |
| `publication/weights/v3_sipakmed_only_seed101/weights/best.pt` | SIPaKMeD Only | 101 | 57 | 5.95 MB | `59389a024247e337cfd08d1eb7b7b353f3ed39e3977170a0620b2699c9faf245` |
| `models/best.pt` | SIPaKMeD (5-class baseline) | -- | -- | 5.97 MB | `9f6245bf9d80870155c7edcb4292e3f87deb989e30e7fc0499ebf17aac9597e1` |

### Audit & Verify Model Weights

Run the automated integrity and safeguard audit anytime:

```bash
python verify_weights.py
```

This verifies that:
- Every `.pt` checkpoint exists and is non-empty.
- Every checkpoint's SHA-256 hash matches `WEIGHTS_MANIFEST.json`.
- `.gitignore` is properly configured so weights are actively tracked in Git.

---

## ⚡ Quickstart Inference

To load any certified checkpoint and run detection on a microscope image:

```python
from ultralytics import YOLO

# 1. Load certified research checkpoint
model = YOLO("publication/weights/v3_combined_seed17/weights/best.pt")

# 2. Predict on image field (frozen development threshold = 0.08, IoU = 0.50)
results = model.predict("microscope_field.jpg", conf=0.08, iou=0.50, imgsz=640)

# 3. Print detections (0: Normal, 1: Abnormal)
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"Cell: {model.names[cls_id]} | Confidence: {conf:.3f} | Box: {box.xyxy[0].tolist()}")
```

---

## 🔬 Reproducibility & Research Protocol

All publication materials, manuscripts, supplementary documents, tables, and verification scripts are centralized in `publication/`.

To re-run the full evaluation and verification pipeline:

```bash
# 1. Run unit and integrity tests
python -m pytest tests

# 2. Verify all release artifacts, split disjointness, and model weights
python publication/scripts/verify_release.py
```

### Dataset Accession
- **SIPaKMeD:** Marina et al., [SIPaKMeD Database](https://www.cs.uoi.gr/~marina/sipakmed.html).
- **APCData:** Mendeley Data, [doi:10.17632/ytd568rh3p.1](https://doi.org/10.17632/ytd568rh3p.1).
- **HMCHH-TCT-CellDet:** Nature Scientific Data, [doi:10.1038/s41597-025-04374-5](https://doi.org/10.1038/s41597-025-04374-5).

---

## 📜 Research Package & Open Artifacts

> **Note:** The full manuscript and supplementary appendix are currently in preparation for formal peer review and journal submission. The open research artifacts, code, checkpoints, and benchmark evidence are structured as follows:

- `publication/figures/`: High-resolution vector PDF and PNG figures (Figures 1–4, Figure S1).
- `publication/tables/`: Complete numerical tables in CSV format (Tables 1–3, Tables S2–S7).
- `publication/weights/`: Certified PyTorch `.pt` checkpoints, args.yaml, and training summaries.
- `publication/reproducibility/`: Automated audit reports, data manifests, and SHA-256 sums.
- `publication/code/`: Complete data preparation, training, evaluation, and visualization scripts.

---

## ⚖️ License & Ethical Declarations

- **Code & Benchmarks:** Released under the MIT License.
- **Model Checkpoints:** Research use only under CC BY-NC 4.0.
- **Author:** abusuraihsakhri (`abusuraihsakhri@gmail.com`)
