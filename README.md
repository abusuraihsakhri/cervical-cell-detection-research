# pap-smear-cyto-algo

## Scope statement

This project trains a YOLO-based cell detector on public cervical cytology datasets (SIPaKMeD, optionally CRIC/Herlev/Mendeley LBC) and tests whether it generalizes across datasets.

**This is a replication-plus-architecture-variant study, not a novel contribution to the core science.** A systematic literature check (via Consensus, Aug 2026) found that cross-dataset generalization for cervical cytology *classification* is an active, already-published area — a 2026 Bioengineering study (Coskun et al.) combined SIPaKMeD + Herlev + CRIC + 416 proprietary multi-center WSIs and reported ResNet50 reaching 91% accuracy / 0.91 macro-F1 on an independent test set. What the literature check did *not* find is a YOLO-based *detection* architecture (as opposed to classification-on-precropped-cells) tested for cross-dataset generalization using public data only, under a hard low-VRAM constraint.

**This project's contribution, stated honestly:** we test whether a lightweight, public-data-only, YOLO-based detection pipeline trained under a 6GB VRAM constraint can approach the ~91% cross-dataset accuracy benchmark set by better-resourced, classification-based, proprietary-data studies. A result well below 91% is not a failure — it is the honest finding about the cost of public-only, low-compute constraints, and is itself worth reporting.

This is **not** a clinical-grade or WHO-compliant diagnostic tool, and does not claim to close an unaddressed research gap. Any cross-dataset numbers reported here should be read alongside, and explicitly compared against, the Coskun et al. 2026 benchmark.

## Hardware target

Single GPU, 6GB VRAM. Primary model: YOLOv8n. See `config.py` for the full rationale and the mandatory training flags (`--half`, batch-size fallback, `--imgsz 640`, `--cache disk`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # then fill in ROBOFLOW_API_KEY
```

## Pipeline

1. **Acquire SIPaKMeD:** `python download_datasets.py`
   Downloads both SIPaKMeD Roboflow mirrors, diffs them, picks the more complete one as primary, and creates a stratified 85/15 train/val split if the export doesn't include one (it didn't, for mirror_a — see `prepare_splits.py`). Also checks whether APCData (Phase 2 dataset, see below) is already on disk and prints the manual download step if not. Writes a dataset status report to `results/dataset_status_report.json` — review this before starting training.

   **Phase 2 second dataset — resolved via research, not the original spec placeholder:** CRIC turned out to use point-click cell-center coordinates, not bounding boxes, and the "Mendeley LBC" dataset the spec's literature review referenced turned out to be classification-labeled only — neither is directly usable for YOLO detection. **APCData** (Mendeley Data, [DOI 10.17632/ytd568rh3p.1](https://data.mendeley.com/datasets/ytd568rh3p/1)) has genuine YOLO-format bounding boxes and was selected instead: 425 images, ~3,619 annotated cells, 6-class Bethesda system (NILM/ASC-US/ASC-H/LSIL/HSIL/SCC). Since its class taxonomy doesn't match SIPaKMeD's 5-class native scheme, cross-dataset comparison uses the 2-class (Normal/Abnormal) roll-up (`config.ROLLUP_APCDATA_6_TO_2`). Mendeley Data has no download API to automate — download it manually from the link above and place the `APCData_YOLO` folder at `data/apcdata/APCData_YOLO`. Full citation trail for all four candidates considered is in `config.PHASE2_DATASET_SOURCES`.

2. **Train (Phase 1):** `python training_pipeline.py --data data/sipakmed/mirror_a/data.yaml`
   YOLOv8n, ImageNet-pretrained init, early stopping on validation mAP@50 (`--epochs 200` is a ceiling, `--patience 30`, not a target). Automatically falls back through batch sizes 16 → 8 → 4 on CUDA OOM. Copy the resulting `best.pt` into `models/` before evaluation.

3. **Evaluate — Section 6a (confusion matrix):**
   `python evaluation/confusion_matrix.py --weights models/best.pt --data data/sipakmed/mirror_a/data.yaml`
   Full 5-class confusion matrix and per-class precision/recall, plus post-hoc 3-class and 2-class roll-ups derived from the same model. Explicitly reports Dyskeratotic ↔ Koilocytotic/Metaplastic confusion.

4. **Evaluate — Section 6b (cross-dataset generalization, Phase 2):**
   `python evaluation/cross_dataset_eval.py --weights models/best.pt --phase1-data data/sipakmed/mirror_a/data.yaml --phase2-data data/apcdata/APCData_YOLO/data.yaml --phase2-dataset-name "APCData"`
   Zero-shot evaluation, no fine-tuning. Reports delta vs. Phase 1 and vs. the Coskun et al. 2026 benchmark. If the second dataset is unavailable, this is recorded as a stated limitation, not omitted.

5. **Evaluate — Section 6c (calibration, Phase 3):**
   `python evaluation/calibration.py --weights models/best.pt --data data/sipakmed/mirror_a/data.yaml`
   Reliability diagram + ECE, with temperature scaling applied and re-reported. Compared against a published classification-based ECE (0.030) as a reference point.

6. **Evaluate — Section 6d (operating point, Phase 3):**
   `python evaluation/pr_curve_threshold.py --weights models/best.pt --data data/sipakmed/mirror_a/data.yaml --target-class Dyskeratotic`
   Full PR curve plus a threshold chosen by an explicit, stated false-negative-to-false-positive cost ratio (default N=10, a placeholder pending Ares's confirmation — see the script for the full rationale).

## Repo structure

```
pap-smear-cyto-algo/
├── config.py
├── download_datasets.py
├── training_pipeline.py
├── evaluation/
│   ├── confusion_matrix.py
│   ├── cross_dataset_eval.py
│   ├── calibration.py
│   └── pr_curve_threshold.py
├── models/                # checkpoints, gitignored except best.pt
├── results/                # generated reports (gitignored runs/, kept json reports)
├── requirements.txt
└── README.md
```

## Non-goals (v1)

- No WSI tiling — single-cell/field images only.
- No HITL active-learning UI.
- No model larger than YOLOv8s without an explicit, documented decision to accept slower training / smaller batch.
- No "clinical," "diagnostic," or "closes a research gap" language anywhere until Sections 6b–6d are complete with documented results, including negative/underwhelming ones.
- No claim of novelty for cross-dataset generalization testing itself.

## Open items (as of last spec update)

- [x] CRIC dataset label verification — RESOLVED via web research: CRIC uses point-click cell-center coordinates, not bounding boxes; not directly usable for YOLO detection. See `config.PHASE2_DATASET_SOURCES`.
- [x] Roboflow API key — confirmed available
- [x] Phase 2 dataset — switched from Mendeley LBC/CRIC to **APCData** (genuine YOLO bounding boxes, see Pipeline step 1 above); requires one manual download step, no automated API for Mendeley Data
- [ ] APCData manual download not yet performed — needed before Phase 2 (`cross_dataset_eval.py`) can run; Phase 1 training is unaffected and can proceed now
- [ ] Whether this project proceeds in parallel with, or instead of, the pending AFB-engine validation/publication work
