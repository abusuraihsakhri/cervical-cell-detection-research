# Critical audit of training, evaluation, and knowledge claims

Date: 19 September 2026

## Verdict

This is a useful exploratory cervical-cell detection project with real training artifacts and valuable investigation of annotation sparsity. It is not scientifically complete, independently validated, or ready to support screening claims. The experimental work is stronger than its current reporting discipline. The most valuable potential contribution is an audit of how annotation completeness, class definitions, and source diversity change detector evaluation, rather than a claim of a novel diagnostic architecture.

Several headline comparisons are invalid because of confirmed class-index errors. Split independence is unresolved and filename coordinates strongly indicate spatial overlap. Dense annotation provenance is insufficient to independently certify expert ground truth. Conclusions about generalization, seed variation, stain invariance, and clinical operating thresholds exceed the experiments.

## Scope and verification

Reviewed training/configuration and dataset preparation code; evaluation and annotation code; saved training CSVs and arguments; JSON results; dense-set manifest and annotation export; README and session claims. Loaded the actual baseline checkpoint to verify its vocabulary. Independently reran baseline inference on all 40 dense fields, correcting only the class mapping, using the existing threshold-sweep matching implementation. Checked train/validation image hashes and source-coordinate filename patterns. Consulted primary dataset/publication sources below.

Original models, labels, evaluators, and reports were not changed. The audit adds this report, a separate verification script, and `critical_audit_evidence_20260919.json`. The corrected metrics still depend on existing dense labels and matcher: they do not certify annotation accuracy or solve split leakage. All other model figures below are saved results, not fresh independent inference runs. No new training, full HMCHH rerun, expert cytology adjudication, or exhaustive literature review was performed.

## Findings ordered by consequence

### 1. Confirmed: abnormal-class mapping corrupts baseline comparisons

The loaded `models/best.pt` vocabulary is:

| ID | Actual class | Correct binary class |
|---|---|---|
| 0 | Dyskeratotic | Abnormal |
| 1 | Koilocytotic | Abnormal |
| 2 | Metaplastic | Normal |
| 3 | Parabasal | Normal |
| 4 | Superficial-Intermediate | Normal |

`evaluation/sweep_operating_thresholds.py:87` and `evaluation/evaluate_hmchh_abnormal_filtered.py:99` instead call IDs 2 and 3 abnormal. They score metaplastic/parabasal predictions as abnormal and discard actual dyskeratotic/koilocytotic predictions from the abnormal category. `annotation/evaluate_dense.py` has another problem: it sends a five-class model and binary ground truth into a matrix without remapping the model classes, then reports diagonal binary metrics. Thus its baseline per-class metrics also are not meaningful.

At confidence 0.04, the independent corrected baseline rerun produced:

| Baseline abnormal metric | Existing threshold-sweep claim | Corrected audit rerun |
|---|---:|---:|
| Precision | 67.57% | 59.50% |
| Recall | 2.77% | 45.98% |
| F1 | 0.0513 | 0.5188 |

Localization metrics are unchanged: precision 68.23%, recall 47.70%, F1 0.5615. The combined model remains promising on the current annotations, but the original abnormal comparison greatly exaggerates its advantage. The baseline HMCHH abnormal result must also be recomputed. Map by checkpoint class names, never assumed numeric order.

### 2. High concern: validation independence is not established

`prepare_splits.py` splits individual files by first-listed class, not by patient, slide, or source image. It calls that class dominant without computing a majority. There are 1,105 training and 193 validation JPGs. Exact SHA-256 overlap was zero, which is useful but does not establish independence.

Six coordinate-bearing filename families occur in both splits: `cell_1`, `cell_2`, `cell_3`, `Cito-4`, `CV1`, and `cv2`. Parsing their coordinates identified 206 train/validation pairs with intersecting source rectangles, involving 45 validation tiles. Eight of the 40 dense fields are among these tiles. These are strong filename-based indicators of spatial leakage, not proof of patient identities or a pixel-registered reconstruction. Verify them against original source images and group all crops from a common source before splitting. Do not describe the current validation set as an independent test cohort.

The Roboflow mirror has 1,298 fields and multiple filename families, whereas the original SIPaKMeD resource describes 966 source cluster images. An explicit source manifest is needed to explain additional images, transforms, annotation authorship, and licensing. The mismatch alone does not establish contamination, but calling the whole mirror original SIPaKMeD is insufficiently documented.

### 3. Dense labels are valuable, but their independence and expert verification are not auditable

The manifest explicitly records C2b as the proposal model, confidence 0.05, and sampling from the existing validation split. It started with 197 sparse ground-truth boxes and 541 model proposals. The purported verified set contains 1,067 boxes: 706 Normal and 361 Abnormal. Comparing export values with the original proposals gives 689 unchanged box/class values, 32 changed values, 346 added boxes, and 17 removed boxes. Export origins are 688 `prediction`, 33 `prediction-changed`, and 346 `manual`; these two summaries measure different properties.

This demonstrates changes to the artifact, not who made them or whether a qualified reviewer exhaustively reviewed the fields. The annotation records contain no metadata outside `result`: no reviewer identity, completion timestamp, or adjudication trail. Retaining proposals is not itself wrong, and `origin=prediction` does not prove a retained box was unreviewed. However, the saved evidence cannot independently substantiate the claim of completed expert verification. The same model that proposed boxes is then scored on them, creating incorporation/anchoring risk unless missed cells were systematically sought and reviewed independently.

Worse, `parse_labelstudio_export` silently falls back to model predictions if annotations are absent, then prints that the images are verified. It defaults unknown label text to Normal and can skip missing images. A scientific evaluation importer must fail closed on unreviewed tasks, unknown classes, missing images, and ambiguous reviewer versions. Preserve authentic exports and document annotator expertise, inclusion rules, box conventions, and an independent second review.

### 4. Confirmed: threshold selection and evaluation reuse the same data

The 65-point sweep selects maximum F1 on the same 40 fields used to report the winning score. Those fields already belong to the validation pool used for model selection. The threshold 0.05 is therefore a development-set optimum, not a validated screening operating point. It also equals the proposal-generation threshold, making annotation independence particularly important; that coincidence is not proof of bias.

The saved C2b result at 0.05 is internally consistent: 687/725 localized detections, 1,067 reference cells, and 281/361 abnormal cells detected. That is 94.76% localization precision, 64.39% localization recall, 95.25% abnormal precision, and 77.84% abnormal recall. It still misses 380 labeled cells, including 80 abnormal cells. These are cell-level development estimates, not slide or patient sensitivity, specificity, or predictive values. Forty fields are not forty patients or necessarily forty slides.

Freeze a threshold on a development set and evaluate once on untouched source-grouped data. Include uncertainty using patient/slide clusters where available, otherwise explicitly limited field-level resampling. Do not treat 1,067 correlated cells as 1,067 independent patients.

### 5. The mixed-source experiment does not demonstrate unseen-domain generalization

C1 is trained on SIPaKMeD alone; C2b is trained on SIPaKMeD plus APCData, including the APCData training split. Improvement on APCData validation from about 24% to 85% class-agnostic recall is evidence that including target-source training data helps. It is not zero-shot generalization to an unseen dataset. Different training durations and validation pools also mean this is not a fully controlled data-source-only ablation.

HMCHH is a more relevant external test. The saved C2b abnormal-only result on 500 fields is 200 true positives among 2,787 predictions and 809 reference abnormal boxes: precision 7.18%, recall 24.72%, F1 0.1112. At this threshold, 609 reference abnormal cells are missed and 2,587 retained predictions do not match an abnormal reference. The result does not establish useful external screening performance. The script takes the first sorted 500 images rather than a prespecified patient-balanced sample.

The primary HMCHH paper confirms abnormal-only annotation, making abnormal filtering appropriate. But removing 11,311 predictions labeled Normal does not prove they are correctly identified normal cells. The report confuses model labels with reference truth. Localization matches also fall from 348 to 200 after filtering; normal predictions include detections on abnormal references. Filtering improves precision partly by sacrificing recall. The assertion that the mechanism is confirmed must be withdrawn.

### 6. Calibration is neither independently validated nor correctly optimized as described

`evaluation/calibration.py:150` claims binary negative-log-likelihood optimization but multiplies its derivative by an extra `scaled * (1 - scaled) * 4`. For `p=sigmoid(z/T)`, the NLL derivative is mean `(p-y)*(-z/T^2)` without that factor. This is a mathematical implementation error, not merely a simplified multiclass method.

Temperature is fitted and ECE reported on the same records. Those correctness labels come from the sparse validation annotations, so detections on unannotated real cells may be incorrectly treated as errors. The recorded improvement from ECE 0.06516 to 0.06341 is an in-sample descriptive change. It does not demonstrate calibrated external probabilities, and detector ECE is conditional on the retained predictions and matching protocol. Recompute against defensible labels, separate fitting and evaluation, and report reliability diagrams, NLL/Brier score, and class/source behavior with uncertainty.

### 7. Replication and statistical claims are unsupported

All four retained training argument files specify `seed: 0` and `deterministic: true`. The original C2 artifact is absent from the retained run directories; its claimed independent seed cannot be checked. Changing the epoch ceiling or workers is not a controlled different-seed replicate. The records do not support a universal recall noise floor of plus/minus 0.03, nor a conclusion that the SIPaKMeD difference is definitively seed noise.

`stain_normalization_benchmark.py` sets `significant_improvement` simply when recall rises by more than 0.03. This is not a statistical significance test. Paired comparisons on the same fields need paired uncertainty; a guessed training noise threshold is not a substitute.

### 8. Stain findings are local observations, not proof of morphology invariance

The tested Reinhard transform improves saved C1 APCData recall by 4.19 percentage points and reduces C2b recall by 23.32 points. This supports reporting a model-dependent effect of this particular inference-time transform. It does not prove multisource training learns morphology instead of color, that all normalization is redundant, or that appearance shift is no longer important. For C2b on APCData, F1 falls only 0.0187 while recall falls 0.2332 and precision increases; the tradeoff matters. APCData is also a trained source for C2b.

Control reference selection, compare multiple references, and distinguish inference-only transformation from models trained with that preprocessing. Avoid making a causal mechanism claim without an experiment that isolates it.

### 9. Training is real and broadly sensible, but documentation is inaccurate

The small YOLO model, mixed precision, bounded epochs, saved arguments/checkpoints, and memory-conscious loading are sensible choices for the available hardware. Saved CSVs show:

| Run | Epochs | Best epoch by mAP50-95 | Best mAP50-95 |
|---|---:|---:|---:|
| SIPaKMeD five-class | 101 | 71 | 0.45439 |
| APCData | 57 | 32 | 0.23036 |
| SIPaKMeD binary | 100 | 86 | 0.43048 |
| Combined binary C2b | 114 | 84 | 0.43787 |

These are different tasks and/or validation populations, so the numbers do not establish which model is scientifically superior. C1 reached its ceiling. There is no strong evidence that more epochs or a larger network is the immediate bottleneck. Fix labels, splits, and evaluation first.

README calls the detector initialization ImageNet-pretrained although the supplied YOLOv8 detection checkpoint should be described by its actual detection pretraining provenance. Training comments say stopping monitors mAP50; check the recorded Ultralytics version's actual fitness criterion. `amp: true` is recorded; treating `half=True` alone as the training mixed-precision control is misleading. Dependencies are open-ended lower bounds, not a reproducible environment lock. The training wrapper reconstructs output paths instead of using the trainer's actual save directory, which can misidentify suffixed runs after retries. It also treats any CUDA error as OOM.

### 10. Knowledge work needs an evidence hierarchy and less certainty

Strengths include noticing incomplete annotation, testing alternatives, preserving negative results, distinguishing morphology from Bethesda labels, and checking annotation formats before dataset use. These are sound research instincts.

Weaknesses include stale README status, conflicting spec/config versus later experiments, unclear source composition, speculative mechanisms stated as established findings, and performance numbers described as clinical without a patient-level endpoint. Sparse labels can harm training as well as evaluation because unannotated cells may contribute misleading background supervision.

The Coskun 2026 publication exists and reports cell-level classification results. Its approximately 91% accuracy is context, not a numerical target directly comparable with detection precision, recall, mAP, or conditional classification accuracy. `cross_dataset_eval.py` still computes diagonals across incompatible class vocabularies and subtracts the publication's benchmark; a caveat does not make those deltas meaningful. Its accuracy also excludes missed and unmatched detections. Remove those deltas, use shared definitions for comparisons, and retain class-agnostic localization only where it answers the intended question.

The claim of a public-data/6 GB performance ceiling is not demonstrated by one lightweight architecture. A constraint is a design condition, not an empirically established upper bound. An AI-assisted search that did not find a matching paper is not a systematic novelty assessment. A traceable literature table needs search dates, queries, primary papers, datasets, splits, task definitions, and limitations. The incomplete fuzzy-ensemble calibration citation should not anchor a quantitative benchmark until identified and checked.

## Recommended order of work

1. Mark existing summaries provisional; correct the baseline class mapping, binary matrix logic, support denominator, and prediction-to-verified fallback. Recompute affected outputs into versioned reports.
2. Reconstruct source membership, resolve coordinate-overlap cases, and rebuild source/slide/patient-grouped splits before making final performance claims. Explain noncanonical mirror files and retain hashes/manifests.
3. Authenticate dense annotation provenance and obtain independent expert review, including an exhaustive search for unproposed cells and adjudication of ambiguous abnormal labels.
4. Separate development, calibration, and final testing. Use one shared name-based class mapping and a documented matching protocol; compare models at common thresholds and at thresholds selected on development data only.
5. Run controlled source ablations with identical budgets, explicitly varied seeds, and an untouched third domain. Report clustered uncertainty, error examples, missed abnormal-cell burden, and false detections per field/slide.
6. Rewrite the synthesis around supported observations. Test patient/slide aggregation and workflow usefulness only if advancing toward a screening claim.

## Overall opinion

Continue the work, but change its immediate objective from claiming completion to establishing trustworthy measurement. More training will not repair an invalid evaluator, a nonindependent split, or unverified reference labels. The strongest prospective paper is a careful, reproducible account of annotation and domain effects in constrained cervical-cell detection. A methods/benchmark study is plausible after these repairs; a clinical validation or novel architecture claim is not supported by the current evidence.

## Primary sources checked

- [Original SIPaKMeD resource](https://www.cs.uoi.gr/~marina/sipakmed.html): original dataset composition; distinguish it from the Roboflow derivative.
- [APCData dataset record](https://data.mendeley.com/datasets/ytd568rh3p/1): source record for the second training/evaluation dataset.
- [HMCHH dataset paper, Scientific Data](https://www.nature.com/articles/s41597-025-04374-5): abnormal-only annotations; 8,037 images derived from 129 slides. Also available through [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11707086/).
- [Coskun et al., Bioengineering 2026](https://www.mdpi.com/2306-5354/13/3/289): cell-level transfer-learning classification comparator. Publication/search records were available; direct full-text retrieval was blocked during this audit, so its full methods were not independently audited.
