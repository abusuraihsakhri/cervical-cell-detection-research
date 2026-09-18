"""
Section 6a — Full confusion matrix + per-class precision/recall (Phase 1).

Reports all 5 classes individually and explicitly calls out confusion
between Dyskeratotic (clinically significant) and Koilocytotic/
Metaplastic, rather than burying it in an aggregate accuracy number.

Also derives the 3-class and 2-class roll-up confusion matrices
post-hoc from this SAME model's predictions, per Section 4 — no
separate models are trained for the coarser class schemes.

Usage:
    python evaluation/confusion_matrix.py --weights models/best.pt --data data/sipakmed/primary/data.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging


def match_predictions_to_gt(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh=0.5):
    """
    Simple greedy IoU matcher between one image's predicted and
    ground-truth boxes. Returns (matched_pred_idx, matched_gt_idx, unmatched_pred, unmatched_gt).
    """
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

    matched_pred, matched_gt = set(), set()
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        best_iou, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            val = iou(pb, gb)
            if val > best_iou:
                best_iou, best_gi = val, gi
        if best_iou >= iou_thresh and best_gi >= 0:
            pairs.append((pi, best_gi))
            matched_pred.add(pi)
            matched_gt.add(best_gi)

    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return pairs, unmatched_pred, unmatched_gt


def build_confusion_matrix(model_path: str, data_yaml: str, conf_thresh: float = config.DEFAULT_INFERENCE_CONF):
    from ultralytics import YOLO
    import yaml

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    n_classes = len(class_names)

    model = YOLO(model_path)

    val_img_dir = Path(data_yaml).parent / data_cfg.get("val", "valid/images")
    val_lbl_dir = val_img_dir.parent / "labels"

    # Predicted class indices come from the model's own vocabulary, which may
    # have more classes than the target data.yaml (e.g. a 5-class SIPaKMeD
    # model zero-shot evaluated against a 1-class binary dataset). Size the
    # matrix to fit whichever vocabulary is larger so pred_classes never
    # indexes out of bounds; the background row/col sits one past the max.
    n_model_classes = len(model.names)
    n_total = max(n_classes, n_model_classes)

    # matrix rows = ground truth, cols = predicted; extra row/col = "background" (missed / spurious)
    cm = np.zeros((n_total + 1, n_total + 1), dtype=int)
    bg = n_total

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
                    # YOLO segmentation format: x1 y1 x2 y2 ... xn yn (polygon, normalized).
                    # Derive an axis-aligned bounding box from the polygon vertices.
                    xs, ys = coords[0::2], coords[1::2]
                    gt_boxes.append((min(xs), min(ys), max(xs), max(ys)))  # xyxy, normalized

        result = model.predict(str(img_path), conf=conf_thresh, verbose=False)[0]
        pred_boxes_xyxyn = result.boxes.xyxyn.cpu().numpy().tolist() if len(result.boxes) else []
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist() if len(result.boxes) else []

        pairs, unmatched_pred, unmatched_gt = match_predictions_to_gt(
            pred_boxes_xyxyn, pred_classes, gt_boxes, gt_classes
        )

        for pi, gi in pairs:
            cm[gt_classes[gi], pred_classes[pi]] += 1
        for pi in unmatched_pred:
            cm[bg, pred_classes[pi]] += 1  # false positive: predicted, no matching GT
        for gi in unmatched_gt:
            cm[gt_classes[gi], bg] += 1  # false negative: missed GT

    return cm, class_names


def per_class_precision_recall(cm: np.ndarray, class_names: list):
    n = len(class_names)
    metrics = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(cm[i, :n].sum())}
    return metrics


def class_agnostic_detection_metrics(cm: np.ndarray):
    """
    Localization-only precision/recall: was *a* cell detected in roughly the
    right place, regardless of which class it was labeled. Useful when GT and
    predicted class vocabularies don't share a taxonomy (e.g. SIPaKMeD's cell
    morphology classes vs. APCData's Bethesda diagnostic categories) — class
    identity is meaningless across such a gap, but "did the detector find the
    cell at all" still isn't.
    cm rows/cols: [class_0, ..., class_n-1, background]. A cell is "found"
    if it landed anywhere in the matched (non-background) submatrix, whether
    or not the predicted class matches the GT class.
    """
    n = cm.shape[0] - 1  # exclude background row/col
    matched = cm[:n, :n].sum()  # any GT matched to any prediction, right or wrong class
    total_gt = cm[:n, :].sum()  # all GT boxes (matched + missed)
    total_pred = cm[:, :n].sum()  # all predicted boxes (matched + spurious)
    recall = float(matched / total_gt) if total_gt > 0 else 0.0
    precision = float(matched / total_pred) if total_pred > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": int(matched),
        "total_gt_boxes": int(total_gt),
        "total_pred_boxes": int(total_pred),
        "note": "Class-agnostic: counts a detection as correct if it localizes a GT cell at IoU>=0.5, regardless of predicted class.",
    }


def report_watch_pairs(cm: np.ndarray, class_names: list):
    """Explicitly surface Dyskeratotic <-> Koilocytotic/Metaplastic confusion, per Section 6a."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    findings = []
    for a, b in config.WATCH_CONFUSION_PAIRS:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        findings.append(
            {
                "pair": f"{a} <-> {b}",
                f"{a}_predicted_as_{b}": int(cm[ia, ib]),
                f"{b}_predicted_as_{a}": int(cm[ib, ia]),
                "clinical_note": (
                    f"{a} is clinically significant; confusion with {b} should be "
                    "reported explicitly, not folded into aggregate accuracy."
                ),
            }
        )
    return findings


def rollup_confusion(cm: np.ndarray, class_names_5: list, mapping: dict, target_classes: list):
    """Derive a coarser confusion matrix from the 5-class one, per Section 4."""
    idx = {name: i for i, name in enumerate(target_classes)}
    n = len(target_classes)
    rolled = np.zeros((n + 1, n + 1), dtype=int)  # +1 for background row/col carried over
    bg_src = len(class_names_5)
    bg_dst = n

    def dst_index(class5_idx):
        if class5_idx == bg_src:
            return bg_dst
        name5 = class_names_5[class5_idx]
        name_target = mapping.get(name5)
        return idx[name_target] if name_target in idx else bg_dst

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            di, dj = dst_index(i), dst_index(j)
            rolled[di, dj] += cm[i, j]
    return rolled


def main():
    setup_logging("confusion_matrix")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE_CONF)
    args = parser.parse_args()

    cm, class_names = build_confusion_matrix(args.weights, args.data, args.conf)
    metrics_5 = per_class_precision_recall(cm, class_names)
    watch = report_watch_pairs(cm, class_names)

    cm_3 = rollup_confusion(cm, class_names, config.ROLLUP_5_TO_3, config.CLASSES_3)
    metrics_3 = per_class_precision_recall(cm_3, config.CLASSES_3)

    cm_2_from_3 = rollup_confusion(cm_3, config.CLASSES_3, config.ROLLUP_3_TO_2, config.CLASSES_2)
    metrics_2 = per_class_precision_recall(cm_2_from_3, config.CLASSES_2)

    out = {
        "5_class": {"confusion_matrix": cm.tolist(), "class_names": class_names, "metrics": metrics_5},
        "3_class_rollup": {"confusion_matrix": cm_3.tolist(), "class_names": config.CLASSES_3, "metrics": metrics_3},
        "2_class_rollup": {"confusion_matrix": cm_2_from_3.tolist(), "class_names": config.CLASSES_2, "metrics": metrics_2},
        "watch_confusion_pairs": watch,
    }

    out_path = config.RESULTS_DIR / "confusion_matrix_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Confusion matrix report written to {out_path}")
    print(json.dumps({"metrics_5_class": metrics_5, "watch_confusion_pairs": watch}, indent=2))


if __name__ == "__main__":
    main()
