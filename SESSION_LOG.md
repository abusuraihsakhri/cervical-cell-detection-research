# Session log — Phase 1-3 training, evaluation, and bug fixes

Covers 2026-08-26 evening through 2026-08-27. Chronological record of what was
done, what was found, and why. See `results/Phase1-3_Training_Evaluation_Report.docx`
for the full narrative report with tables and interpretation; this log is the
terse action-by-action trail.

## 2026-08-26

- **Phase 1 training completed.** YOLOv8n trained on SIPaKMeD (mirror_a split,
  1,105 train / 193 valid, 5 classes). Early-stopped at epoch 101/200 (patience
  30), best checkpoint from epoch 71, ~28.4 min wall clock on an RTX 3060
  Laptop GPU. Training-time mAP@50-95 = 0.454. `models/best.pt` staged.
- User's PC restarted unexpectedly after this point.

## 2026-08-27 — session resumed after restart

- **Verified nothing was lost.** Checked file timestamps, `results.csv`,
  `phase1_sipakmed_training_summary.json`, and log files directly rather than
  trusting memory. All Phase 1 artifacts were complete and timestamped before
  the restart — training had already finished when the PC went down.

- **Ran Phase 2 zero-shot cross-dataset eval** (`evaluation/cross_dataset_eval.py`
  against APCData). Got an internally inconsistent result: SIPaKMeD
  in-distribution "accuracy" = 1.0 alongside macro-F1 = 0.0006, with per-class
  support of 0 for 4/5 classes. Investigated instead of reporting as-is.

- **Bug #1 found and fixed — ground-truth label misparsing.**
  `evaluation/confusion_matrix.py` parsed every label line as YOLO bbox format
  (`class cx cy w h`), but SIPaKMeD's labels are segmentation polygons
  (`class x1 y1 x2 y2 ... xn yn`). The code was treating the first two polygon
  vertices as if they were box width/height, so IoU matching against
  predictions almost never succeeded. Fixed in three scripts
  (`confusion_matrix.py`, `calibration.py`, `pr_curve_threshold.py`): detect
  4-value (bbox) vs. >4-value (polygon) label lines, and for polygons derive an
  axis-aligned box from the min/max of the polygon vertices. APCData's labels
  were already in bbox format, so only the SIPaKMeD-side numbers were affected.

- **Re-ran all four evaluation scripts post-fix.** Sane numbers:
  - Phase 1 (SIPaKMeD): accuracy 0.975 (among matched detections), macro-F1
    0.531. Per-class recall 0.51-0.75, precision 0.40-0.56.
  - Phase 2 (APCData zero-shot): accuracy 0.169, macro-F1 0.037 (later
    re-interpreted — see below).
  - Calibration: raw ECE 0.065, temperature-scaled ECE 0.063 (T=0.981, near
    identity — little systematic miscalibration to correct).
  - Dyskeratotic PR curve: cost-weighted (10x FN cost vs. FP cost) operating
    point at threshold 0.111, precision 0.417 / recall 0.624, vs. recall
    0.51-0.51 at the previous default of 0.25.

- **Report written**: `results/Phase1-3_Training_Evaluation_Report.docx`
  (protocol, chronology, results tables, interpretation, ranked
  recommendations).

- User asked to act on the 5 recommendations one by one:

  1. **Class-agnostic cross-dataset metric.** A defensible SIPaKMeD-morphology
     ↔ APCData-Bethesda class mapping wasn't judged achievable without
     clinical input, so added `class_agnostic_detection_metrics()` to
     `confusion_matrix.py` — localization-only precision/recall/F1 (IoU>=0.5,
     any predicted class counts as a match). Wired into
     `cross_dataset_eval.py`'s `phase2_cross_dataset.class_agnostic_detection`.
     Result: precision holds (~0.45-0.46) across SIPaKMeD and APCData, but
     recall collapses on APCData (0.647 -> 0.118) — real domain shift in
     detection ability, not just a taxonomy artifact.

  2. **Dyskeratotic cost ratio (N=10).** Decision: keep as documented
     placeholder in `pr_curve_threshold.py`; no code change. Flagged for
     clinical confirmation before the 0.111 threshold is used beyond this
     study.

  3. **Recall as the binding constraint.** Decision: lower the default
     inference confidence threshold rather than retrain. Added
     `config.DEFAULT_INFERENCE_CONF = 0.111` with a rationale comment;
     switched `confusion_matrix.py` and `cross_dataset_eval.py`'s `--conf`
     defaults to it. Re-ran the SIPaKMeD confusion matrix at the new default —
     recall gains held across all 5 classes (not just Dyskeratotic), at a
     fairly uniform precision cost.

  4. **HMCHH-TCT cross-dataset eval.** Crashed on first attempt —
     **Bug #2**: `build_confusion_matrix()` sized its confusion matrix using
     only the target dataset's class count (HMCHH-TCT is 1-class binary, so a
     2x2 matrix), but predictions come from the 5-class SIPaKMeD-trained
     model, so a predicted class index >=2 had nowhere to go (`IndexError`).
     Fixed by sizing the matrix to `max(target_classes, model_classes) + 1`.
     Re-ran (1,077 validation images, took several minutes — backgrounded).
     Result: opposite failure mode from APCData — the model *over-fires*
     wildly (26,315 predicted boxes vs. 2,054 GT boxes, ~13:1), recall 0.468
     but precision only 0.037. Suggests HMCHH-TCT's images make background
     regions look cell-like to this model (staining/resolution/artifact
     differences), a distinct domain-shift failure mode from APCData's
     under-detection.

  5. **Label-format sanity check.** Not built as new tooling — validated
     retroactively: both item 1's and item 4's investigations surfaced their
     own instance of this same bug class. Documented at both fix sites and in
     the report instead of adding a separate automated checker.

- **Report updated**: appended an addendum section to
  `results/Phase1-3_Training_Evaluation_Report.docx` covering all 5 follow-up
  actions, their results tables, and interpretation.

## 2026-09-10 — status report, environment repair, domain-shift analysis

- **Wrote `results/Project_Status_and_Roadmap.docx`** — full retrospective across
  the whole project plus a tiered forward roadmap. Supersedes and extends
  `Phase1-3_Training_Evaluation_Report.docx`.

- **Found two artifacts lost since 31 Aug.**
  - `models/best.pt` was gone (only `.gitkeep` left). Restored from
    `results/runs/phase1_sipakmed/weights/best.pt`. Every README command
    pointed at the missing file.
  - `results/cross_dataset_eval_report.json` held only the HMCHH-TCT run; the
    APCData numbers had been overwritten. Recovered from
    `logs/cross_dataset_eval_20260827_003056.log` into
    `results/cross_dataset_eval_apcdata_cervical_cytology_cells.json`.

- **Bug #3 found and fixed — evaluation output overwrite.**
  `cross_dataset_eval.py` wrote to one fixed filename, so each new target
  dataset destroyed the previous one's report. Now derives a per-dataset
  filename from `--phase2-dataset-name`, with an `--out` override.

- **Environment was broken.** `import torch` failed with `OSError WinError 126`
  on `cudnn_cnn64_9.dll`. The named DLL existed; 4 of its ~8 sibling cuDNN
  DLLs (`cudnn_ops64_9`, `cudnn_adv64_9`, `cudnn_engines_precompiled64_9`,
  `cudnn_heuristic64_9`) were missing. Force-reinstalled torch 2.10.0+cu130.
  Verified `torch.cuda.is_available()` is True and `best.pt` loads under
  ultralytics 8.4.31.

- **Quantified the domain shift** (was only a hypothesis before).
  `results/dataset_image_statistics.json`, `results/dataset_domain_shift.png`.
  SIPaKMeD vs APCData: green-minus-red +0.105 vs -0.004 (stain), contrast
  0.136 vs 0.099, texture energy 0.00283 vs 0.00143, median box area fraction
  0.0082 vs 0.0179 (~2.2x), cells/image 4.86 vs 8.47, 14.9% vs 0% of images
  more than 35% near-white. Multi-factor shift: colour, contrast, texture,
  object scale, field density, composition.

- **Measured prediction density on both full validation splits**
  (`results/prediction_density_analysis.json`). At conf 0.111 the model emits
  2.17 boxes per GT box on SIPaKMeD and 0.54 on APCData — a 4x drop in firing
  rate, confirming under-firing on the full split.

- **Rendered GT-vs-prediction overlays** (`results/predictions_sipakmed.png`,
  `results/predictions_apcdata.png`) — the top-ranked open action. This
  surfaced the session's most important finding:

  **SIPaKMeD is sparsely annotated relative to the cells present.** Images
  with 20+ clearly visible cells routinely carry 1-2 GT boxes, and the model's
  "false positives" land on visible unlabelled cells. Consequences:
  - In-distribution precision (0.33-0.40) is a lower bound, not a measurement.
  - The HMCHH-TCT precision of 0.037 is substantially an annotation-protocol
    artifact. `prepare_hmchh.py` had already documented that only abnormal
    cells get boxes there; that note was never connected to the precision
    figure. A model detecting all cells, scored against abnormal-only labels,
    scores near-zero precision regardless of quality.
  - The "two opposite failure modes" framing survives only in weakened form.
    APCData under-firing is well supported and now explained; HMCHH-TCT
    over-firing is confounded and cannot be cleanly attributed to domain shift.

- **Report addendum (Section 11)** written covering all of the above, with the
  three figures embedded and a revised priority list.

## Files touched 2026-09-10

- `evaluation/cross_dataset_eval.py` — per-dataset output filenames + `--out`.
- `models/best.pt` — restored.
- `results/Project_Status_and_Roadmap.docx` — new, then extended with Section 11.
- `results/dataset_image_statistics.json`, `results/dataset_domain_shift.png` — new.
- `results/prediction_density_analysis.json`, `results/predictions_*.png` — new.
- `results/cross_dataset_eval_apcdata_cervical_cytology_cells.json` — recovered.
- `results/cross_dataset_eval_hmchh_tct_binary.json` — copy under new naming.

## New open items (2026-09-10)

- Quantify SIPaKMeD annotation completeness (count real cells vs labelled cells
  on 20-30 images). Until this is known, every precision figure in the project
  is uninterpretable. Now the highest-priority open item.
- Re-evaluate HMCHH-TCT counting only abnormal-class predictions, or drop it
  from precision claims.
- Stain normalization and scale-jitter augmentation both promoted on the
  strength of the measured shift.

## 2026-09-10 (later) — annotation completeness resolved

Acted on the previous entry's top open item. Answer arrived from three
independent directions and is worse than the suspicion.

- **Documentary evidence, decisive.** `data/sipakmed/mirror_a/README.dataset.txt`
  states: "The SIPaKMeD Database consists of 4049 images of isolated cells that
  have been manually extracted from 966 cluster cell images of Pap smear
  slides. This dataset is a subset of the SIPaKMeD. Only the single-cell images
  are rearranged for purpose of experiment."

  4049 / 966 = **4.19 annotated cells per field**. Measured GT density in our
  splits is 4.86 (train) / 4.54 (valid). They agree.

  SIPaKMeD is a *classification* dataset of hand-picked isolated cells
  retrofitted into detection boxes. Not exhaustive detection ground truth, and
  the selection is *biased* — cells were picked for being cleanly isolated and
  unambiguously classifiable.

- **Independent counting** (`results/annotation_completeness.json`,
  `results/annotation_check_*.png`). Built a nucleus counter that does not use
  the trained model. Three detector designs tried, two rejected on visual
  inspection (documented in the `detect_nuclei` docstring): Otsu + connected
  components segmented cytoplasm sheets; DoG-with-scale-filter fired on
  cytoplasm texture in dense fields. Kept: local maxima of the haematoxylin
  channel. Result over 30 images each — SIPaKMeD 135 GT boxes vs 1,860 detected
  cells (**7.3%** annotated); APCData 253 vs 1,641 (**15.4%**). The counter
  overcounts (it also finds leukocyte nuclei), so true fractions are higher;
  the robust part is that APCData is ~2x better annotated and neither is close
  to complete.

- **Manual verification** (`results/manualcount_cell_2_9600_400_10100_.png`).
  One validation field, rendered full size and counted by eye: ~55-70 squamous
  epithelial cells with visible nuclei, **2 GT boxes** (~3%). Both annotated
  cells are isolated and sit in clear space while dozens of cells in
  overlapping sheets carry no label. The selection bias is visible in one image.

### What this invalidates

- **All precision, F1, macro-F1 and ECE figures in the project.** Computed
  against labels covering under a tenth of the cells present. Most "false
  positives" are correct detections of unlabelled cells.
- HMCHH-TCT precision 0.037 is doubly confounded (sparse annotation + that
  dataset's abnormal-only protocol). Should not be quoted at all.
- The class-agnostic localization metric does not escape it — same denominator
  problem in its precision term.
- **Recall, mAP and the firing-rate results stand.** Recall against a sparse
  label set still answers its own question honestly.

### What this means for the model

Not just an evaluation problem. The model was *trained* on these labels, so it
learned that ~4 cells per field are targets and the other ~50 are background —
an implicit "annotatable cell" concept meaning isolated, well-separated,
unambiguous. APCData fields are denser (8.47 vs 4.86 cells/image), i.e. exactly
what SIPaKMeD taught it to ignore. **The APCData under-firing may be label
selection bias rather than the appearance shift measured earlier.** The
appearance shift is real but is no longer the only credible explanation, and
may not be the main one.

### Revised priorities

1. Stop quoting precision/F1/ECE until a densely annotated eval set exists.
   Report recall, mAP and qualitative cross-dataset findings.
2. Densely annotate 30-50 SIPaKMeD fields (every cell boxed), evaluation only.
   Now the highest-value work available.
3. Re-run the evaluation suite against it; the delta is itself reportable.
4. Reconsider SIPaKMeD as a detection training set — or switch to APCData, a
   densely re-annotated subset, or classification-on-crops.
5. **Demote** stain normalization and augmentation (promoted in the previous
   entry). Eliminate the label problem first, or a label fix gets credited to
   a stain fix.

### Files touched

- `results/annotation_completeness.json` — new (counts, method, rejected
  detector designs, provenance quote, manual check).
- `results/annotation_check_sipakmed.png`, `results/annotation_check_apcdata.png` — new.
- `results/manualcount_cell_2_9600_400_10100_.png` — new.
- `results/Project_Status_and_Roadmap.docx` — Section 12 added; Section 11.4 and
  the executive summary now point at it.

## 2026-09-10 (later still) — experiments A and C: label density confirmed as the cause

Ran two training experiments to test Section 12's conclusion. Both support it;
C produces the project's first clearly positive result.

### Epoch budget (question answered)

Epochs were never the constraint — a full run costs 5-30 min on this card.

| Run | Data | Ceiling | Ran to | Best ep | Best mAP50-95 |
|---|---|---|---|---|---|
| Phase 1 | SIPaKMeD 5-class | 200 | 101 (early stop) | 71 | 0.4544 |
| A | APCData 6-class | 100 | 57 (early stop) | 32 | 0.2304 |
| C1 | SIPaKMeD 2-class | 100 | 100 (hit ceiling) | 86 | 0.4305 |
| C2 | Combined 2-class | 100 | 90 (interrupted) | 66 | 0.4557 |

Phase 1 hit 95% of best by ep42, peaked at 71, then val_cls_loss rose
1.463 -> 1.907 (overfitting). **Use 100/patience 25 for 5- and 6-class; 150 for
2-class (C1 peaked at 86 and hit the ceiling); 150-200/patience 40 if heavily
augmented.** `training_pipeline.py` now takes `--epochs` / `--patience`.

### Experiment A — same imagery, different training labels

Trained YOLOv8n on APCData (354 imgs, ~2x label density), compared against the
SIPaKMeD-trained model. Both measured on both val sets, class-agnostic
(IoU>=0.5, conf 0.111) since class counts differ (5 vs 6) with no defensible
mapping. `results/firing_rate_matrix.json`.

| Trained on | Fire SIPaKMeD | Recall SIPaKMeD | Fire APCData | Recall APCData |
|---|---|---|---|---|
| SIPaKMeD | 2.17 | 0.765 | 0.55 | 0.215 |
| APCData | **4.18** | 0.608 | 1.35 | 0.641 |

(fire = predicted boxes per GT box)

**The two SIPaKMeD columns hold imagery constant** — same 193 images, so stain,
contrast, texture, resolution, density all controlled. The APCData-trained model
still fires **1.9x more often**. Appearance shift cannot explain that; the only
difference is what the training labels taught each model to treat as a target.
Label-density hypothesis confirmed by design of the test, not by argument.

Transfer is asymmetric: SIPaKMeD->APCData recall 0.215, APCData->SIPaKMeD 0.608
from 1/3 the data. The APCData model is the *weaker* model by its own training
metric (0.230 vs 0.454 mAP50-95), so that confound runs against the result.

### Experiment C — multi-source training

Both datasets rolled up to Normal/Abnormal via existing config mappings
(new script `prepare_combined_2class.py`, hard-links images so 3 variants
don't cost 3 copies).

| Model | Fire SIPaKMeD | Recall SIPaKMeD | Fire APCData | Recall APCData |
|---|---|---|---|---|
| C1 SIPaKMeD only (1105 img) | 2.58 | 0.807 | 0.68 | 0.240 |
| C2 combined (1459 img) | 1.96 | 0.749 | 2.01 | **0.860** |

**Adding 354 APCData images raises APCData recall 0.240 -> 0.860 (3.6x) for a
0.058 cost on SIPaKMeD.** First clearly positive result in this project.

Firing ratios matter as much: C1 is 2.58/0.68 (4x asymmetry), Phase 1 is
2.17/0.55, C2 is 1.96/2.01. **C2 is the only model tested that behaves
consistently across both domains** rather than collapsing on one.

### Honest notes

- C2 was killed at ep90/100 by a system low-memory reaper during ultralytics'
  close-mosaic step (also killed the driver shell). Cost nothing: C2's best is
  ep66, C1's is ep86, both before the mosaic-off window (ep91+), so the
  comparison stays symmetric. No summary JSON for C2 — process never reached
  that line.
- C1 and C2 have *different val sets* (193 vs 193+63 images), so their mAP
  figures are not comparable. That is why the comparison uses the firing-rate
  matrix, which runs every model over identical splits.
- Precision deliberately omitted from all tables above (Section 12). Recorded
  in the JSON as `precision_lower_bound` with the caveat inline.
- Single seed per config. The 0.240->0.860 gain is far too large to be seed
  noise; the 0.058 SIPaKMeD loss is not, and shouldn't be treated as real
  without repeats.

### What this changes

- **Appearance shift demoted.** Real, but not the main driver.
- **Stain normalization withdrawn as top intervention.** Untested; multi-source
  training was tested and worked.
- **"Does not transfer" refined**: a model trained on *one* sparsely-labelled
  dataset doesn't transfer. Trained on two, it does. The barrier was the
  training data, not the architecture or the 6 GB budget.
- Dense annotation of a SIPaKMeD eval set: unchanged priority, now more urgent —
  every result here is recall/firing-rate precisely because precision is still
  unmeasurable.

### Files touched

- `training_pipeline.py` — `--epochs` / `--patience` overrides; `data` recorded
  in the summary JSON.
- `prepare_combined_2class.py` — new. Builds sipakmed_only / apcdata_only /
  combined 2-class datasets from the config roll-ups.
- `evaluation/firing_rate_matrix.py` — new. Class-agnostic model x dataset
  comparison; `--device cpu` so it can run while a training holds VRAM.
- `results/firing_rate_matrix.json`, `results/combined_2class_dataset_stats.json`,
  `results/expA_apcdata_training_summary.json`,
  `results/expC1_sipakmed_2class_training_summary.json` — new.
- `results/runs/expA_apcdata/`, `expC1_sipakmed_2class/`, `expC2_combined_2class/` — new.
- `results/Project_Status_and_Roadmap.docx` — Section 13 added; executive
  summary points at it.

### Next

- Repeat C2 across 2-3 seeds to firm up the SIPaKMeD-side cost.
- Densely annotate 30-50 SIPaKMeD fields (still the blocking item for precision).
- Re-run C2 to completion (ep 91-100) if a clean 100-epoch run is wanted for
  the record; not expected to change the result.
- Only then test stain normalization, against a baseline that is no longer
  confounded by label sparsity.

## 2026-09-10 (C2 rerun) — multi-source result replicates; the SIPaKMeD "cost" was noise

Reran the combined model from scratch at a 150-epoch ceiling
(`expC2b_combined_150`, patience 30, workers 2). Early-stopped at ep114,
best mAP50-95 0.4379 @ ep84.

### Cleared first: the previous C2 was hung, not dead

GPU showed 4311/6144 MiB used with no training running. Cause: PID 7264 (the
killed C2 run) was still alive and hung, plus 12 orphaned dataloader workers,
all from 09:08-09:09. Killed all 13 — freed 2.9 GB VRAM and 3.2 GB RAM
(free RAM had been 2.4 GB of 15.2, which is what tripped the reaper).

**Root cause of the original hang:** ultralytics' `close_mosaic` (default 10)
tears down the dataloader and builds a new one at ep91 of a 100-epoch run,
allocating fresh workers *before* the old ones are reaped, so peak memory
briefly doubles at that exact epoch. With 2.4 GB free the respawn blocked
indefinitely instead of failing loudly, and `results.csv` kept looking healthy
throughout. Mitigations: `--workers 2` (new flag) halves the spike, and the
C2b status monitor treats a stall as a failure rather than only watching for
crash text — the previous monitor reported healthy progress through 15 min of
a dead process.

### Replication result

| Run | Recall SIPaKMeD | Recall APCData | Fire SIPaKMeD | Fire APCData |
|---|---|---|---|---|
| Phase 1 (SIPaKMeD 5-class) | 0.765 | 0.215 | 2.17 | 0.55 |
| C1 (SIPaKMeD only, 2-class) | 0.807 | 0.240 | 2.58 | 0.68 |
| C2 (combined) | 0.749 | **0.860** | 1.96 | 2.01 |
| C2b (combined, repeat) | 0.765 | **0.854** | 2.20 | 2.06 |

**CONFIRMED** — the APCData recall gain replicates to within 0.006 (0.860 /
0.854 vs C1's 0.240). A 3.6x gain landing twice from independent runs is real.
Balanced firing replicates too (2.20 / 2.06 vs C1's 4x asymmetry).

**RETRACTED** — the 0.058 SIPaKMeD recall "cost" was seed noise, as flagged in
the previous entry. C2b scores 0.765 on SIPaKMeD, *identical* to Phase 1, and
the three 2-class runs spread 0.807 / 0.749 / 0.765 — a range of 0.058 that
entirely contains the supposed effect.

Correct statement: **multi-source training buys a 3.6x recall gain on the
second dataset at no established cost on the first.**

**Calibration point for the whole project:** run-to-run recall variation on this
setup is roughly ±0.03. Any difference smaller than that, anywhere in these
results, must not be interpreted from a single run.

### Epoch question closed

Best epochs across five runs: 71, 32, 86, 66, 84. C2b tested the 150 ceiling
directly and early-stopped at 114. **150 is confirmed headroom, not a target**;
C1's ceiling-hit at 100 was the outlier. `training_pipeline.py` now takes
`--epochs`, `--patience` and `--workers`.

### Files touched

- `training_pipeline.py` — added `--workers`; `workers` recorded in summary JSON.
- `results/runs/expC2b_combined_150/` — new run.
- `results/expC2b_combined_150_training_summary.json` — new.
- `results/firing_rate_matrix.json` — regenerated with all 5 models.
- `results/Project_Status_and_Roadmap.docx` — Section 13.5 (replication) added,
  13.6 renumbered, epoch table and honest-notes updated.

### Next (unchanged priority order)

1. Densely annotate 30-50 SIPaKMeD fields — still the blocking item; every
   result here is recall/firing-rate because precision remains unmeasurable.
2. A third combined seed only if the SIPaKMeD-side question matters; two seeds
   already settle the APCData gain.
3. Stain normalization — test only against a baseline no longer confounded by
   label sparsity, and expect ±0.03 noise, so a single run will not resolve a
   small effect.

## Files touched this session

- `evaluation/confusion_matrix.py` — label-format fix (bbox vs. polygon);
  confusion-matrix sizing fix (mismatched GT/model class counts); added
  `class_agnostic_detection_metrics()`; `--conf` default now
  `config.DEFAULT_INFERENCE_CONF`.
- `evaluation/calibration.py` — label-format fix.
- `evaluation/pr_curve_threshold.py` — label-format fix.
- `evaluation/cross_dataset_eval.py` — added `class_agnostic_detection` block
  to the Phase 2 report; `--conf` default now `config.DEFAULT_INFERENCE_CONF`.
- `config.py` — added `DEFAULT_INFERENCE_CONF = 0.111` with rationale.
- `results/confusion_matrix_report.json`, `results/cross_dataset_eval_report.json`,
  `results/calibration_report.json`, `results/pr_curve_dyskeratotic_report.json`
  — regenerated with fixes and new default threshold.
- `results/Phase1-3_Training_Evaluation_Report.docx` — created, then extended
  with an addendum.

## Open items / not yet done

- No defensible APCData(Bethesda) <-> SIPaKMeD(morphology) class mapping has
  been built; the class-agnostic metric is a workaround, not a resolution.
- N=10 Dyskeratotic cost ratio remains clinically unvalidated.
- HMCHH-TCT's severe over-firing (precision 0.037) has not been root-caused
  beyond "likely a staining/resolution/artifact domain-shift effect" — worth
  visually inspecting a few HMCHH-TCT predictions if this dataset matters
  going forward.
- Per-class PR-curve/operating-point analysis has only been done for
  Dyskeratotic; the global 0.111 default was extrapolated from that one class.
