"""
Operating threshold sweep and PR-curve analysis on the verified dense SIPaKMeD evaluation set.

Evaluates models across confidence thresholds from 0.01 to 0.70 to determine:
  1. Full Precision-Recall curves (Overall and Abnormal-specific).
  2. Optimal F1 operating point.
  3. Clinical screening operating points (high sensitivity, high precision, cost-weighted).
  4. Generates high-resolution multi-panel figures and JSON reports.

Usage:
    python evaluation/sweep_operating_thresholds.py
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging


def calculate_iou(boxA, boxB):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def load_ground_truth(val_yaml: Path):
    """Load ground truth annotations from verified dataset."""
    import yaml
    cfg = yaml.safe_load(val_yaml.read_text(encoding="utf-8"))
    val_img_dir = val_yaml.parent / cfg.get("val", "images")
    val_lbl_dir = val_img_dir.parent / "labels"

    gt_data = {}
    img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    for img_path in img_files:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # normalized cx, cy, w, h -> x1, y1, x2, y2
                    cx, cy, w, h = map(float, parts[1:5])
                    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                    boxes.append({"cls": cls_id, "bbox": (x1, y1, x2, y2)})
        gt_data[img_path.name] = {"path": img_path, "boxes": boxes}
    return gt_data, cfg.get("names", ["Normal", "Abnormal"])


def collect_raw_predictions(model_path: Path, gt_data: dict, min_conf: float = 0.01):
    """Run model inference once at min_conf and collect all candidate predictions."""
    from ultralytics import YOLO
    config.verify_model_checksum(model_path)
    model = YOLO(str(model_path))

    predictions = {}
    for img_name, info in gt_data.items():
        res = model.predict(str(info["path"]), conf=min_conf, imgsz=640, verbose=False)[0]
        preds = []
        if len(res.boxes) > 0:
            boxes_xyxyn = res.boxes.xyxyn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes_xyxyn, confs, clss):
                # Map classes: if model has 5 classes (SIPaKMeD 5-class), map:
                # 0 (Superficial-Intermediate) -> 0 (Normal)
                # 1 (Parabasal) -> 0 (Normal)
                # 2 (Koilocytotic) -> 1 (Abnormal)
                # 3 (Dyskeratotic) -> 1 (Abnormal)
                # 4 (Metaplastic) -> 0 (Normal)
                mapped_cls = cls_id
                if len(model.names) == 5:
                    mapped_cls = 1 if cls_id in (2, 3) else 0
                preds.append({
                    "bbox": tuple(box),
                    "conf": float(conf),
                    "cls": int(mapped_cls),
                    "orig_cls": int(cls_id),
                })
        predictions[img_name] = preds
    return predictions


def evaluate_at_threshold(gt_data: dict, predictions: dict, conf_thresh: float, iou_thresh: float = 0.5):
    """Evaluate detections at a specific confidence threshold."""
    total_gt = 0
    total_gt_abnormal = 0
    total_gt_normal = 0

    total_pred = 0
    total_pred_abnormal = 0

    tp_class_agnostic = 0
    tp_abnormal = 0
    tp_normal = 0

    for img_name, info in gt_data.items():
        gt_boxes = info["boxes"]
        preds = [p for p in predictions[img_name] if p["conf"] >= conf_thresh]

        total_gt += len(gt_boxes)
        total_gt_normal += sum(1 for g in gt_boxes if g["cls"] == 0)
        total_gt_abnormal += sum(1 for g in gt_boxes if g["cls"] == 1)

        total_pred += len(preds)
        total_pred_abnormal += sum(1 for p in preds if p["cls"] == 1)

        # Greedy match for class-agnostic
        matched_gt_agnostic = set()
        for p in preds:
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gt_boxes):
                if gi in matched_gt_agnostic:
                    continue
                iou = calculate_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thresh and best_gi >= 0:
                matched_gt_agnostic.add(best_gi)
                tp_class_agnostic += 1

        # Class-specific match (Abnormal)
        gt_abn_indices = [gi for gi, g in enumerate(gt_boxes) if g["cls"] == 1]
        pred_abn = [p for p in preds if p["cls"] == 1]
        matched_gt_abn = set()
        for p in pred_abn:
            best_iou, best_gi = 0.0, -1
            for gi in gt_abn_indices:
                if gi in matched_gt_abn:
                    continue
                iou = calculate_iou(p["bbox"], gt_boxes[gi]["bbox"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thresh and best_gi >= 0:
                matched_gt_abn.add(best_gi)
                tp_abnormal += 1

    # Class-agnostic metrics
    prec = tp_class_agnostic / total_pred if total_pred > 0 else 1.0
    rec = tp_class_agnostic / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Abnormal metrics
    prec_abn = tp_abnormal / total_pred_abnormal if total_pred_abnormal > 0 else 1.0
    rec_abn = tp_abnormal / total_gt_abnormal if total_gt_abnormal > 0 else 0.0
    f1_abn = 2 * prec_abn * rec_abn / (prec_abn + rec_abn) if (prec_abn + rec_abn) > 0 else 0.0

    return {
        "conf": float(conf_thresh),
        "total_pred": int(total_pred),
        "total_gt": int(total_gt),
        "tp": int(tp_class_agnostic),
        "fp": int(total_pred - tp_class_agnostic),
        "fn": int(total_gt - tp_class_agnostic),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "abnormal_total_pred": int(total_pred_abnormal),
        "abnormal_total_gt": int(total_gt_abnormal),
        "abnormal_tp": int(tp_abnormal),
        "abnormal_precision": float(prec_abn),
        "abnormal_recall": float(rec_abn),
        "abnormal_f1": float(f1_abn),
    }


def sweep_model(model_name: str, weights_path: Path, gt_data: dict, thresholds: np.ndarray):
    """Run sweep across all thresholds for a single model."""
    print(f"Sweeping thresholds for: {model_name}...")
    raw_preds = collect_raw_predictions(weights_path, gt_data, min_conf=float(thresholds[0]))
    results = []
    for thresh in thresholds:
        metrics = evaluate_at_threshold(gt_data, raw_preds, conf_thresh=thresh)
        results.append(metrics)
    return results


def find_optimal_points(sweep_results: list):
    """Extract key clinical and technical operating points."""
    # 1. Best F1
    best_f1_pt = max(sweep_results, key=lambda x: x["f1"])
    best_abn_f1_pt = max(sweep_results, key=lambda x: x["abnormal_f1"])

    # 2. High Sensitivity Screening (target >= 60% and >= 70% recall)
    sens_candidates = [r for r in sweep_results if r["recall"] >= 0.50]
    best_screen_pt = max(sens_candidates, key=lambda x: x["precision"]) if sens_candidates else sweep_results[0]

    # 3. High Precision Confirmation (target >= 95% precision)
    prec_candidates = [r for r in sweep_results if r["precision"] >= 0.95]
    best_confirm_pt = max(prec_candidates, key=lambda x: x["recall"]) if prec_candidates else sweep_results[-1]

    # 4. Cost-weighted operating point (Cost = FP + 10 * FN)
    best_cost_pt = min(sweep_results, key=lambda x: x["fp"] + 10 * x["fn"])

    return {
        "max_f1_point": best_f1_pt,
        "max_abnormal_f1_point": best_abn_f1_pt,
        "high_sensitivity_screening": best_screen_pt,
        "high_precision_confirmation": best_confirm_pt,
        "cost_weighted_10x_fn": best_cost_pt,
    }


def plot_results(all_sweeps: dict, output_png: Path):
    """Plot comprehensive 4-panel publication figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    colors = {
        "Exp C2b (Combined 2-class)": "#1b365d", # Navy
        "Exp C1 (SIPaKMeD 2-class)": "#d9534f",  # Red
        "Phase 1 (SIPaKMeD 5-class)": "#5bc0de", # Cyan
    }

    # Panel A: Metrics vs Threshold for Exp C2b
    axA = axes[0, 0]
    c2b_sweep = all_sweeps.get("Exp C2b (Combined 2-class)", [])
    confs = [r["conf"] for r in c2b_sweep]
    axA.plot(confs, [r["precision"] for r in c2b_sweep], label="Precision (Class-Agnostic)", color="#28a745", linewidth=2.2)
    axA.plot(confs, [r["recall"] for r in c2b_sweep], label="Recall (Class-Agnostic)", color="#007bff", linewidth=2.2)
    axA.plot(confs, [r["f1"] for r in c2b_sweep], label="F1-Score", color="#1b365d", linewidth=2.5, linestyle="--")
    axA.plot(confs, [r["abnormal_f1"] for r in c2b_sweep], label="Abnormal F1-Score", color="#fd7e14", linewidth=2.0, linestyle=":")
    axA.axvline(0.111, color="gray", linestyle="-.", alpha=0.7, label="Default Conf (0.111)")
    axA.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
    axA.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    axA.set_title("(A) Exp C2b: Performance vs. Confidence Threshold", fontsize=12, fontweight="bold", color="#1b365d")
    axA.set_xlim(0.01, 0.65)
    axA.set_ylim(0.0, 1.02)
    axA.legend(loc="lower left", frameon=True)

    # Panel B: PR Curve (Overall Cell Detection)
    axB = axes[0, 1]
    for name, sweep in all_sweeps.items():
        recs = [r["recall"] for r in sweep]
        precs = [r["precision"] for r in sweep]
        axB.plot(recs, precs, label=name, color=colors.get(name, "black"), linewidth=2.2)
    axB.set_xlabel("Recall (Localization IoU >= 0.5)", fontsize=11, fontweight="bold")
    axB.set_ylabel("Precision", fontsize=11, fontweight="bold")
    axB.set_title("(B) Precision-Recall Curve: All Epithelial Cells", fontsize=12, fontweight="bold", color="#1b365d")
    axB.set_xlim(0.0, 0.70)
    axB.set_ylim(0.5, 1.02)
    axB.legend(loc="lower left", frameon=True)

    # Panel C: Abnormal Cell PR Curve
    axC = axes[1, 0]
    for name, sweep in all_sweeps.items():
        recs_abn = [r["abnormal_recall"] for r in sweep]
        precs_abn = [r["abnormal_precision"] for r in sweep]
        axC.plot(recs_abn, precs_abn, label=name, color=colors.get(name, "black"), linewidth=2.2)
    axC.set_xlabel("Recall (Abnormal Cells)", fontsize=11, fontweight="bold")
    axC.set_ylabel("Precision (Abnormal Cells)", fontsize=11, fontweight="bold")
    axC.set_title("(C) Precision-Recall Curve: High-Risk Abnormal Cells", fontsize=12, fontweight="bold", color="#1b365d")
    axC.set_xlim(0.0, 0.70)
    axC.set_ylim(0.5, 1.02)
    axC.legend(loc="lower left", frameon=True)

    # Panel D: Total Detections & False Alarms vs Threshold
    axD = axes[1, 1]
    fps = [r["fp"] for r in c2b_sweep]
    tps = [r["tp"] for r in c2b_sweep]
    total_gt = c2b_sweep[0]["total_gt"] if c2b_sweep else 1067
    axD.plot(confs, tps, label=f"True Detections (out of {total_gt} GT)", color="#28a745", linewidth=2.2)
    axD.plot(confs, fps, label="False Positives (Spurious Boxes)", color="#dc3545", linewidth=2.2)
    axD.axvline(0.111, color="gray", linestyle="-.", alpha=0.7, label="Default Conf (0.111)")
    axD.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
    axD.set_ylabel("Count (Across 40 Slides)", fontsize=11, fontweight="bold")
    axD.set_title("(D) Detection Volume & False Alarms (Exp C2b)", fontsize=12, fontweight="bold", color="#1b365d")
    axD.set_xlim(0.01, 0.65)
    axD.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"Saved publication PR curve figure to: {output_png}")


def main():
    setup_logging("sweep_operating_thresholds")
    val_yaml = config.DATA_DIR / "dense_eval" / "sipakmed_dense40_verified" / "data.yaml"
    if not val_yaml.exists():
        print(f"ERROR: {val_yaml} does not exist.", file=sys.stderr)
        sys.exit(1)

    gt_data, class_names = load_ground_truth(val_yaml)
    print(f"Loaded {len(gt_data)} verified fields with {sum(len(v['boxes']) for v in gt_data.values())} total cells.")

    models_to_test = [
        ("Exp C2b (Combined 2-class)", config.RESULTS_DIR / "runs" / "expC2b_combined_150" / "weights" / "best.pt"),
        ("Exp C1 (SIPaKMeD 2-class)", config.RESULTS_DIR / "runs" / "expC1_sipakmed_2class" / "weights" / "best.pt"),
        ("Phase 1 (SIPaKMeD 5-class)", config.MODELS_DIR / "best.pt"),
    ]

    thresholds = np.linspace(0.01, 0.65, 65)
    all_sweeps = {}
    summary_report = {
        "verified_fields": len(gt_data),
        "verified_ground_truth_cells": sum(len(v["boxes"]) for v in gt_data.values()),
        "thresholds_tested": [float(t) for t in thresholds],
        "models": {},
    }

    for name, path in models_to_test:
        if path.exists():
            sweep = sweep_model(name, path, gt_data, thresholds)
            all_sweeps[name] = sweep
            optimal_pts = find_optimal_points(sweep)
            summary_report["models"][name] = {
                "weights_path": str(path),
                "optimal_operating_points": optimal_pts,
                "detailed_sweep": sweep,
            }

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<28} | {'Best F1':<9} | {'Opt Conf':<8} | {'Precision':<10} | {'Recall':<8} | {'Abnormal F1':<11}")
    print("=" * 90)
    for name, data in summary_report["models"].items():
        opt = data["optimal_operating_points"]["max_f1_point"]
        abn_opt = data["optimal_operating_points"]["max_abnormal_f1_point"]
        print(f"{name:<28} | {opt['f1']:<9.4f} | {opt['conf']:<8.2f} | {opt['precision']:<10.4f} | {opt['recall']:<8.4f} | {abn_opt['abnormal_f1']:<11.4f}")
    print("=" * 90)

    # Save outputs
    json_path = config.RESULTS_DIR / "dense_operating_threshold_sweep.json"
    json_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"Saved threshold sweep metrics to: {json_path}")

    png_path = config.RESULTS_DIR / "pr_curve_dense_verified.png"
    plot_results(all_sweeps, png_path)


if __name__ == "__main__":
    main()
