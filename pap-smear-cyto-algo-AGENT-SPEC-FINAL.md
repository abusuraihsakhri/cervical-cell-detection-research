# Pap Smear Cytopathology YOLO Detection — Agent Build Spec (Final)

**Give this entire file to your coding agent as the project brief. It supersedes all earlier drafts in this thread.**

**Owner:** Ares
**Hardware:** single GPU, 6GB VRAM
**Predecessor repos to reuse patterns from (not code-copy blindly):** `blast-detection-algo`, `pretrained-tb-afb-detection-engine`

---

## 0. Honest scope statement — put this near-verbatim in the eventual README

This project trains a YOLO-based cell detector on public cervical cytology datasets (SIPaKMeD, optionally CRIC/Herlev) and tests whether it generalizes across datasets.

**This is a replication-plus-architecture-variant study, not a novel contribution to the core science.** A systematic check of the literature (via Consensus, Aug 2026) found:

- Cross-dataset generalization for cervical cytology classification is an **active, already-published area**. A 2026 Bioengineering study (Coskun et al.) combined SIPaKMeD + Herlev + CRIC + 416 proprietary multi-center WSIs and reported ResNet50 reaching **91% accuracy / 0.91 macro-F1** on an independent test set, explicitly showing that combining diverse sources mitigates scanner-induced domain shift.
- Several other papers (fuzzy ensemble frameworks, multi-domain hybrid models) report some form of cross-dataset or external validation.
- **What is NOT covered in the literature found:** a YOLO-based *detection* architecture (as opposed to classification-on-precropped-cells) tested for cross-dataset generalization, using **public data only** (no proprietary WSIs), under a hard low-VRAM constraint.

**Therefore, state the project's contribution honestly as:** "We test whether a lightweight, public-data-only, YOLO-based detection pipeline trained under a 6GB VRAM constraint can approach the ~91% cross-dataset accuracy benchmark set by better-resourced, classification-based, proprietary-data studies." A result well below 91% is not a failure — it is the honest finding about the cost of public-only, low-compute constraints, and is itself worth reporting.

**Never write:** "clinical grade," "WHO-compliant," "diagnostic tool," or imply this closes an unaddressed research gap. **Always cite** the Coskun et al. 2026 benchmark when reporting your own cross-dataset numbers.

---

## 1. What went wrong in the two predecessor repos (do not repeat)

| Issue | Fix in this project |
|---|---|
| Single dataset, no external validation | Cross-dataset test included (Section 6), correctly framed as replication, not discovery |
| Single confidence threshold reported, no PR curve | Full PR curve + justified operating point required (Section 6d) |
| No calibration check | ECE + reliability diagram required (Section 6c) |
| Overclaimed "clinical grade" language on unvalidated models | Banned; see Section 0 |
| Epoch count chosen arbitrarily (150 vs 20 across two repos, no rationale) | Early stopping on validation mAP; epoch number is a ceiling, not a target (Section 4) |

---

## 2. Hardware-driven model choice (6GB VRAM)

| Model | Params | Approx VRAM @ batch16, imgsz640, FP16 | Fits 6GB? |
|---|---|---|---|
| YOLOv8n | 3.2M | ~2-3GB | Yes — comfortable, room for larger batch |
| YOLOv8s | 11.2M | ~4-5GB | Yes, tight — batch 8-12 |
| YOLOv8m+ | 26M+ | 7GB+ | No |

**Use YOLOv8n as primary.** Reuse the `yolov8n.pt` ImageNet-pretrained checkpoint pattern already used in `pretrained-tb-afb-detection-engine` rather than training from random init. Escalate to YOLOv8s only if Phase 1 baseline accuracy proves insufficient — do not skip straight to larger models.

**Mandatory training flags:**
```
--half              # FP16, ~40-50% memory reduction
--batch 16          # drop to 8, then 4, if OOM
--imgsz 640         # don't raise to 1280 during training on this card
--cache disk        # avoid full in-memory cache as dataset grows
--workers 4
```

---

## 3. Datasets and acquisition status

| Dataset | Status | Access method |
|---|---|---|
| **SIPaKMeD** | **Confirmed downloadable** | Roboflow Universe, two mirrors: `ik-zu-quan-o9tdm/sipakmed-ioflq` and `sipakmed/sipakmed-t9emb`. Use `pip install roboflow` + API key (`.env` pattern: `ROBOFLOW_API_KEY=`, same as `blast-detection-algo`'s `download_datasets.py`). YOLOv8 export format supported natively. |
| **CRIC Cervix** | **Unconfirmed — labels not yet verified by Ares** | Original source: CRIC Searchable Image Database (cric.com.br). Agent must attempt access and verify the download actually contains cell-level bounding-box + class annotations (not just image-level metadata) before treating Phase 2 as unblocked. If what's obtained is unlabeled or only partially labeled, **stop and report back — do not silently substitute another dataset or proceed as if labels exist.** |
| **Herlev** | Available, optional | University of Ioannina source / mirrors. Smaller (917 images), older. Optional third source or pretraining supplement. |
| **Mendeley LBC** | Available, optional | Used in several papers found in literature check (Manna et al. 2021, Yaman et al. 2022) as a second public dataset alongside SIPaKMeD — worth checking as a CRIC alternative if CRIC access stalls, since it's already used this way in published work. |

**Agent action order:**
1. Download both SIPaKMeD Roboflow mirrors, diff class counts/splits, pick the more complete one as primary.
2. Attempt CRIC access; **verify annotation format with a sample file before writing any code that assumes bounding boxes exist.**
3. If CRIC is blocked or unlabeled, fall back to Mendeley LBC as the second dataset (precedent exists in literature for this substitution) and document the substitution and why.
4. Report dataset status back to Ares before starting Phase 2 training — do not proceed silently on an assumption.

---

## 4. Class mapping

SIPaKMeD native classes: Superficial-Intermediate, Parabasal, Koilocytotic, Dyskeratotic, Metaplastic.

- **Primary target: 5-class (native).** Matches most published benchmarks for direct comparison.
- Derive 3-class (Normal/Benign/Abnormal) and 2-class (Normal/Abnormal) as **post-hoc roll-ups of the same trained model's confusion matrix** — do not train three separate models for this.

---

## 5. Epoch strategy — stopping rule, not a fixed number

No dataset-independent correct epoch count exists. Evidence: AFB engine peaked at epoch 150/150 (cap reached, no plateau confirmed); blast detector peaked at epoch 10/20 (10 epochs wasted). Neither number transfers to this dataset.

```
--epochs 200      # ceiling / safety cap only
--patience 30     # stop if val mAP@50 hasn't improved in 30 epochs
```

**[Guessing, unverified on actual hardware]:** given SIPaKMeD's small size (4,049 images) and reported ceiling effects in the literature (98-99%+ accuracy across multiple published methods), expect convergence around epoch 40-80 on YOLOv8n. Agent should log actual per-epoch wall-clock time from the first 3 epochs and report a corrected total-time estimate rather than trusting any number in this document — none of these timing figures have been benchmarked on this specific GPU.

---

## 6. Evaluation — this is the actual deliverable, not an afterthought

### 6a. Full confusion matrix + per-class precision/recall (Phase 1)
Report all 5 classes individually. Explicitly call out confusion between Dyskeratotic (clinically significant) and Koilocytotic/Metaplastic (can be morphologically similar) — do not bury this in an aggregate accuracy number.

### 6b. Cross-dataset generalization (Phase 2 — framed per Section 0)
- Train on SIPaKMeD only.
- Zero-shot evaluate on CRIC (or Mendeley LBC fallback) with no fine-tuning.
- Report the accuracy/mAP/recall delta.
- **Explicitly compare against the Coskun et al. 2026 benchmark (91% accuracy / 0.91 macro-F1)**, noting that benchmark used proprietary multi-center WSI data in addition to public sources — a lower number here is expected and should be discussed as "the public-data-only, YOLO-detection ceiling," not treated as a failure.
- If second dataset access fails entirely, document this explicitly as a stated limitation rather than omitting the section.

### 6c. Calibration (Phase 3)
Reliability diagram + Expected Calibration Error (ECE) on the SIPaKMeD validation set. Apply temperature scaling if raw YOLO confidence is poorly calibrated (expected, per general object-detector literature) and re-report ECE after correction. Note: at least one paper in the literature check (fuzzy ensemble, 2026) already reports ECE (0.030) for a classification-based approach — cite it as the comparison point, since this may be the first ECE reported for a *detection*-based approach in this specific task.

### 6d. Operating point selection (Phase 3)
Full PR curve, not a single threshold. Justify the chosen operating point with an explicit, stated cost ratio (e.g., "a missed Dyskeratotic cell is treated as N times costlier than a false-positive flag" — state N and the reasoning, even if simplified).

---

## 7. Repo structure

```
pap-smear-cyto-algo/
├── config.py                    # hyperparameters, paths, class mapping (Section 4)
├── download_datasets.py         # SIPaKMeD (Roboflow) + CRIC/Mendeley LBC acquisition (Section 3)
├── training_pipeline.py         # YOLOv8n training, early stopping (Section 5)
├── evaluation/
│   ├── confusion_matrix.py      # Section 6a
│   ├── cross_dataset_eval.py    # Section 6b
│   ├── calibration.py           # Section 6c
│   └── pr_curve_threshold.py    # Section 6d
├── models/                      # checkpoints, gitignore except final best.pt
├── requirements.txt             # ultralytics, roboflow, scikit-learn, matplotlib
└── README.md                    # Section 0 scope statement goes here, verbatim in spirit
```

---

## 8. Explicit non-goals (v1)

- No WSI tiling — single-cell/field images only; no public pyramidal WSI Pap dataset exists at this scale to justify it yet.
- No HITL active-learning UI — defer to a later phase once a validated baseline + generalization result exists.
- No model larger than YOLOv8s without an explicit decision to accept slower training / smaller batch.
- No "clinical," "diagnostic," or "closes a research gap" language anywhere until Sections 6b-6d are complete with documented results, including negative/underwhelming ones.
- Do not claim novelty for cross-dataset generalization testing itself — cite Coskun et al. 2026 and frame this project's contribution narrowly and honestly, per Section 0.

---

## 9. Open items requiring Ares's input before/during build (agent should surface these, not guess)

- [ ] CRIC dataset label verification (sample annotation file not yet reviewed)
- [ ] Confirm Roboflow API key availability
- [ ] Confirm whether Mendeley LBC is an acceptable fallback if CRIC access fails
- [ ] Decide whether this project proceeds in parallel with, or instead of, the pending AFB-engine validation/publication work (unresolved as of this spec's writing)
