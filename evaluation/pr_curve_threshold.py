"""
Section 6d — Operating point selection (Phase 3).

Full PR curve, not a single threshold. Justifies the chosen operating
point with an explicit, stated cost ratio: "a missed Dyskeratotic cell
is treated as N times costlier than a false-positive flag."

The default N below is a placeholder the agent must NOT treat as
clinically validated — Ares should confirm or override it. It exists
so the script runs end-to-end and produces a concrete, arguable number
rather than leaving the cost ratio unspecified.

Usage:
    python evaluation/pr_curve_threshold.py --weights models/best.pt --data data/sipakmed/primary/data.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging

# Placeholder cost ratio: how many times costlier a missed Dyskeratotic
# detection (false negative) is treated as, relative to one false-positive
# flag. This is a simplified, stated assumption per spec Section 6d
# ("even if simplified") — NOT a clinically derived figure. Surface this
# prominently in any report and ask Ares to confirm/override.
FN_TO_FP_COST_RATIO_DYSKERATOTIC = 10
COST_RATIO_RATIONALE = (
    "A missed Dyskeratotic (clinically significant, precancer-associated) cell "
    "is treated as 10x costlier than one false-positive flag, on the reasoning "
    "that a false positive costs a cytotechnologist a few seconds of extra "
    "review, while a false negative risks a missed early cancer signal. This "
    "N=10 is a stated, simplified placeholder per spec Section 6d, not a "
    "clinically validated cost model — Ares should confirm or override it "
    "before this operating point is used for anything beyond this study."
)


def collect_class_confidences(model_path: str, data_yaml: str, target_class_name: str, iou_thresh: float = 0.5):
    from ultralytics import YOLO
    import yaml

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    target_idx = class_names.index(target_class_name)

    model = YOLO(model_path)
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

    scored = []  # (confidence, is_true_positive) for the target class only
    total_gt_positives = 0

    img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    for img_path in img_files:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls != target_idx:
                    continue
                coords = list(map(float, parts[1:]))
                if len(coords) == 4:
                    # YOLO bbox format: cx cy w h (normalized)
                    cx, cy, w, h = coords
                    gt_boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                else:
                    # YOLO segmentation format: x1 y1 x2 y2 ... xn yn (polygon, normalized)
                    xs, ys = coords[0::2], coords[1::2]
                    gt_boxes.append((min(xs), min(ys), max(xs), max(ys)))
        total_gt_positives += len(gt_boxes)

        result = model.predict(str(img_path), conf=0.001, verbose=False)[0]
        if len(result.boxes) == 0:
            continue
        pred_boxes = result.boxes.xyxyn.cpu().numpy().tolist()
        pred_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        pred_confs = result.boxes.conf.cpu().numpy().tolist()

        matched_gt = set()
        for pb, pc, pconf in zip(pred_boxes, pred_classes, pred_confs):
            if pc != target_idx:
                continue
            best_iou, best_gi = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                val = iou(pb, gb)
                if val > best_iou:
                    best_iou, best_gi = val, gi
            is_tp = best_iou >= iou_thresh
            if is_tp:
                matched_gt.add(best_gi)
            scored.append((pconf, is_tp))

    return scored, total_gt_positives


def compute_pr_curve(scored, total_gt_positives, n_thresholds=100):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    curve = []
    for t in thresholds:
        tp = sum(1 for conf, is_tp in scored if conf >= t and is_tp)
        fp = sum(1 for conf, is_tp in scored if conf >= t and not is_tp)
        fn = total_gt_positives - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        curve.append({"threshold": float(t), "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})
    return curve


def select_operating_point(curve, fn_cost_ratio):
    """
    Pick the threshold minimizing a simple weighted cost:
        cost = fn_cost_ratio * FN + 1 * FP
    per the stated cost ratio in Section 6d.
    """
    best = min(curve, key=lambda p: fn_cost_ratio * p["fn"] + p["fp"])
    return best


def main():
    setup_logging("pr_curve_threshold")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-class", default="Dyskeratotic")
    parser.add_argument("--fn-cost-ratio", type=float, default=FN_TO_FP_COST_RATIO_DYSKERATOTIC)
    args = parser.parse_args()

    print(f"Collecting {args.target_class} confidence scores across validation set...")
    scored, total_gt = collect_class_confidences(args.weights, args.data, args.target_class)
    curve = compute_pr_curve(scored, total_gt)
    operating_point = select_operating_point(curve, args.fn_cost_ratio)

    out = {
        "target_class": args.target_class,
        "fn_cost_ratio_used": args.fn_cost_ratio,
        "cost_ratio_rationale": COST_RATIO_RATIONALE,
        "full_pr_curve": curve,
        "selected_operating_point": operating_point,
    }
    out_path = config.RESULTS_DIR / f"pr_curve_{args.target_class.lower()}_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nPR curve + operating point report written to {out_path}")
    print(f"Selected threshold: {operating_point['threshold']:.3f} "
          f"(precision={operating_point['precision']:.3f}, recall={operating_point['recall']:.3f})")
    print(f"\nCost ratio rationale (N={args.fn_cost_ratio}):\n{COST_RATIO_RATIONALE}")


if __name__ == "__main__":
    main()
