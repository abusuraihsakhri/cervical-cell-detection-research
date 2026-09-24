"""
Section 6c — Calibration (Phase 3).

Reliability diagram + Expected Calibration Error (ECE) on the SIPaKMeD
validation set. Applies temperature scaling if raw YOLO confidence is
poorly calibrated (expected per general object-detector literature),
and re-reports ECE after correction.

Temperature is fitted on a development dataset and evaluated on a separate
image-disjoint dataset. Source-group independence remains a protocol requirement.
No unverified classification-calibration benchmark comparison is made.

Usage:
    python evaluation/calibration.py --weights models/best.pt --data fit/data.yaml --eval-data evaluation/data.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging
from evaluation.taxonomy import native_index_map


def collect_confidence_correctness(model_path: str, data_yaml: str, conf_thresh: float = 0.05, iou_thresh: float = 0.5):
    """
    For every predicted box above a low confidence floor, record
    (confidence, is_correct) where is_correct means it matched a
    ground-truth box of the same class at iou_thresh.
    """
    from ultralytics import YOLO
    import yaml

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    model = YOLO(model_path)
    class_map = native_index_map(model.names, data_cfg["names"])

    val_img_dir = Path(data_yaml).parent / data_cfg.get("val", "valid/images")
    val_lbl_dir = val_img_dir.parent / "labels"

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    records = []  # (confidence, correct)
    img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    for img_path in img_files:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        gt_boxes, gt_classes = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                gt_classes.append(cls)
                if len(coords) == 4:
                    # YOLO bbox format: cx cy w h (normalized)
                    cx, cy, w, h = coords
                    gt_boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                else:
                    # YOLO segmentation format: x1 y1 x2 y2 ... xn yn (polygon, normalized)
                    xs, ys = coords[0::2], coords[1::2]
                    gt_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        result = model.predict(str(img_path), conf=conf_thresh, verbose=False)[0]
        if len(result.boxes) == 0:
            continue
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()
        pred_classes = [class_map[int(i)] for i in result.boxes.cls.cpu().numpy()]
        pred_confs = result.boxes.conf.cpu().numpy().tolist()

        matched_gt = set()
        for pb, pc, pconf in zip(pred_boxes, pred_classes, pred_confs):
            best_iou, best_gi = 0.0, -1
            for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
                if gi in matched_gt or gc != pc:
                    continue
                val = iou(pb, gb)
                if val > best_iou:
                    best_iou, best_gi = val, gi
            correct = best_iou >= iou_thresh
            if correct:
                matched_gt.add(best_gi)
            records.append((pconf, correct))

    return records


def expected_calibration_error(records, n_bins=10):
    confs = np.array([r[0] for r in records])
    corrects = np.array([1.0 if r[1] else 0.0 for r in records])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_report = []
    n = len(confs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() == 0:
            bins_report.append({"range": [float(lo), float(hi)], "count": 0, "avg_conf": None, "accuracy": None})
            continue
        bin_conf = confs[mask].mean()
        bin_acc = corrects[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bins_report.append(
            {
                "range": [float(lo), float(hi)],
                "count": int(mask.sum()),
                "avg_confidence": float(bin_conf),
                "accuracy": float(bin_acc),
            }
        )
    return float(ece), bins_report


def fit_temperature_scale(records, lr=0.01, iters=500):
    """
    Fit a single scalar temperature T minimizing NLL on the
    confidence-vs-correctness records, treating confidence as a
    sigmoid-like probability. This is a simplified 1D temperature fit
    appropriate for post-hoc detector confidence calibration, not a
    full multi-class Platt/temperature scaling over logits (YOLO
    doesn't expose pre-sigmoid class logits per-box uniformly across
    versions) — documented here as a known simplification.
    """
    confs = np.clip(np.array([r[0] for r in records]), 1e-6, 1 - 1e-6)
    corrects = np.array([1.0 if r[1] else 0.0 for r in records])
    logits = np.log(confs / (1 - confs))

    # Optimize log-temperature with stable binary cross entropy.
    from scipy.optimize import minimize_scalar
    def nll(log_temperature):
        z = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, z) - corrects * z))
    if not len(records):
        raise ValueError("Cannot calibrate empty records")
    optimum = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    if not optimum.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(optimum.x))


def apply_temperature(records, T):
    logits = np.log(np.clip(np.array([r[0] for r in records]), 1e-6, 1 - 1e-6) / (1 - np.clip(np.array([r[0] for r in records]), 1e-6, 1 - 1e-6)))
    scaled_confs = 1 / (1 + np.exp(-logits / T))
    return list(zip(scaled_confs.tolist(), [r[1] for r in records]))


def main():
    setup_logging("calibration")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Calibration-fit data")
    parser.add_argument("--eval-data", required=True, help="Separate held-out calibration evaluation data")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    print("Collecting confidence/correctness pairs from validation predictions...")
    import hashlib, yaml
    def image_hashes(yaml_path):
        cfg = yaml.safe_load(Path(yaml_path).read_text())
        root = Path(yaml_path).parent / cfg.get("val", "valid/images")
        files = [p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}]
        if not files:
            raise ValueError(f"No images in {root}")
        return {hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if image_hashes(args.data) & image_hashes(args.eval_data):
        raise ValueError("Fit/evaluation images overlap; grouped source independence must also be checked")
    records = collect_confidence_correctness(args.weights, args.data)
    if not records:
        print("ERROR: no predictions collected — check weights/data paths.", file=sys.stderr)
        sys.exit(1)

    if Path(args.data).resolve() == Path(args.eval_data).resolve():
        raise ValueError("Fit and evaluation datasets must be separate")
    evaluation_records = collect_confidence_correctness(args.weights, args.eval_data)
    if not evaluation_records:
        raise ValueError("No held-out evaluation predictions")
    raw_ece, raw_bins = expected_calibration_error(evaluation_records, args.bins)
    print(f"Raw ECE: {raw_ece:.4f}")

    T = fit_temperature_scale(records)
    scaled_records = apply_temperature(evaluation_records, T)
    scaled_ece, scaled_bins = expected_calibration_error(scaled_records, args.bins)
    print(f"Temperature-scaled ECE (T={T:.3f}): {scaled_ece:.4f}")

    out = {
        "raw_ece": raw_ece,
        "raw_reliability_bins": raw_bins,
        "fitted_temperature": T,
        "scaled_ece": scaled_ece,
        "scaled_reliability_bins": scaled_bins,
        "n_predictions": len(evaluation_records),
        "n_fit_predictions": len(records),
        "fit_data": args.data,
        "evaluation_data": args.eval_data,
        "status": "Requires independent source-grouped fit/evaluation sets and complete reference annotations",
        "method_note": (
            "Temperature fit is a simplified 1D scaling over YOLO's exported "
            "box confidence (treated as a sigmoid probability), not a full "
            "multi-class logit temperature scaling. Documented as a known "
            "simplification, not a full calibration-literature implementation."
        ),
    }
    out_path = config.RESULTS_DIR / "calibration_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nCalibration report written to {out_path}")


if __name__ == "__main__":
    main()
