# Model card: cervical-cell detectors (v3, six checkpoints)

## Model

YOLOv8n object detectors (3.0 million parameters) with two classes, `0: Normal` and `1: Abnormal`. Each was initialized from the Ultralytics COCO-pretrained `yolov8n.pt` (SHA-256 `f59b3d83…3b36`) and trained with Ultralytics 8.4.31 and PyTorch 2.10.0.

| Checkpoint | Training data | Seed |
|---|---|---|
| `v3_sipakmed_only_seed17/weights/best.pt` | SIPaKMeD derivative | 17 |
| `v3_sipakmed_only_seed43/weights/best.pt` | SIPaKMeD derivative | 43 |
| `v3_sipakmed_only_seed101/weights/best.pt` | SIPaKMeD derivative | 101 |
| `v3_combined_seed17/weights/best.pt` | SIPaKMeD derivative + APCData | 17 |
| `v3_combined_seed43/weights/best.pt` | SIPaKMeD derivative + APCData | 43 |
| `v3_combined_seed101/weights/best.pt` | SIPaKMeD derivative + APCData | 101 |

SHA-256 hashes, best epochs and training durations are listed in `../reproducibility/release_verification.json` (section `checkpoints`) and in `../SHA256SUMS.txt`. Every run folder contains the exact Ultralytics `args.yaml`, the per-epoch `results.csv` and a training summary.

## Intended use

Research on measurement, reproducibility and dataset effects in cervical-cell detection. The operating threshold each model used in the paper is `threshold_from_development` in `../results/v3_*_evaluation.json`.

## Out-of-scope use

These models are not medical devices, and they have not been validated for screening, diagnosis or any clinical decision. They were not evaluated at slide or patient level. Their abnormal/normal labels come from morphology classes (SIPaKMeD) and Bethesda categories (APCData) that were merged into a binary task; they do not output Bethesda categories.

## Training data

Component-disjoint splits of publicly available images; see the manuscript Methods and `../reproducibility/data_manifest.json`. No dataset images are distributed with this package.

## Evaluation

See the manuscript Results, Tables 2–3 and the per-field results in `../results/`. Performance on data from other laboratories, scanners or preparation methods is expected to differ. External performance on HMCHH was substantially lower than on held-out fields from the training sources.

## Known limitations

The dense reference used for threshold selection and evaluation was annotated by one reviewer starting from model proposals. Patient identities were unavailable, so independence rests on image-overlap components and specimen codes.

## Loading

```python
from ultralytics import YOLO
model = YOLO('v3_combined_seed17/weights/best.pt')
result = model.predict('field.jpg', imgsz=640, conf=THRESHOLD, iou=0.7, max_det=300)[0]
```
