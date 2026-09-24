# Reproducing the study

The steps below rebuild every number in the manuscript from the public datasets. Each step checks the hashes of frozen inputs and refuses to run if they changed.

## Environment

Python 3.12, PyTorch 2.10.0 (CUDA 13.0), Ultralytics 8.4.31, NumPy 2.5.2, SciPy 1.18.1, OpenCV 4.13.0. `requirements.txt` lists the direct dependencies; `reproducibility/environment_snapshot.json` records the exact versions used. Training used one NVIDIA RTX 3060 Laptop GPU (6 GB). Deterministic mode was on, but bit-identical weights across different GPUs, drivers or library versions are not guaranteed. Evaluation from the released weights is deterministic.

## Data

Download the public sources and place them as follows:

| Source | Location expected by the scripts |
|---|---|
| SIPaKMeD derivative (Roboflow export, YOLO polygon format) | `data/sipakmed/mirror_a/{train,valid}` |
| APCData V1, YOLO labels (doi:10.17632/ytd568rh3p.1) | `data/apcdata/APCData cervical cytology cells/`, then `python prepare_apcdata.py` |
| HMCHH-TCT-CellDet (doi:10.6084/m9.figshare.27901206) | images and XML placed in `data/hmchh/` (configured in `prepare_hmchh.py`) |
| Dense reference (40 fields) | `data/dense_eval/sipakmed_dense40_verified/` rebuilt from the released Label Studio export with `annotation/evaluate_dense.py` |

`reproducibility/data_manifest.json` lists every field with its SHA-256, so you can confirm that your downloads match ours.

## Order of execution

```
python prepare_hmchh.py                               # prepared HMCHH validation images
python prepare_hmchh_abnormal_reference.py            # abnormal-cell-only external reference
python publication/scripts/detect_field_overlap.py    # overlap evidence (about 2 h on 16 cores; exhaustive within source)
python publication/scripts/verify_overlap_photometric.py  # pixel check of every confirmed overlap
python prepare_research_v3.py                         # component-grouped splits and dense split
python evaluation/research_v3.py --freeze             # hash-locked evaluation protocol (must precede training)
bash publication/scripts/run_pipeline_v3.sh           # train 6 models, evaluate, synthesize, figures, verify
python publication/scripts/build_docx.py              # Word versions of manuscript and supplement
```

To evaluate the released weights without retraining, copy `weights/v3_*` into `results/research_v3/runs/` and start from `python evaluation/research_v3.py`.

## Verification

`python publication/scripts/verify_release.py` re-derives the dataset from the manifest and checks:

- split files equal a fresh conversion of the source labels, with no boxes dropped;
- no overlap component, exact duplicate or confirmed overlapping pair crosses splits;
- no dense field or its component was used in training;
- the dense and HMCHH references match their frozen hashes, and HMCHH contains no organism labels;
- each checkpoint's seed, data split, class names and training arguments are consistent, and every reported result came from the released weights.

The report is written to `reproducibility/release_verification.json`. The test suite (`python -m pytest tests`) covers class mapping, fail-closed annotation import, calibration, matching, overlap grouping and the HMCHH conversion.
