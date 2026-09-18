"""
Evaluates trained models on HMCHH-TCT with protocol-aligned Abnormal-Only filtering.

HMCHH-TCT annotates exclusively abnormal/dysplastic cells; normal squamous cells
are unlabelled. Earlier zero-shot evaluation scored 0.037 precision because every
detected normal cell was penalized as a false positive.

This script evaluates both:
  1. Unfiltered (all predicted cells counted against abnormal labels) -> reproduces 0.037 baseline.
  2. Protocol-Aligned Filter (only predicted Abnormal cells counted) -> isolates true abnormal transfer.

Usage:
    python evaluation/evaluate_hmchh_abnormal_filtered.py
    python evaluation/evaluate_hmchh_abnormal_filtered.py --max-images 300
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


def match_boxes(preds, gts, iou_thresh=0.5):
    """Greedy IoU matching."""
    matched_gt = set()
    tp = 0
    for p in preds:
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = calculate_iou(p, g)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def load_hmchh_val(hmchh_dir: Path, max_images: int = None):
    """Load HMCHH validation image paths and ground truth boxes."""
    val_img_dir = hmchh_dir / "valid" / "images"
    val_lbl_dir = hmchh_dir / "valid" / "labels"

    img_files = sorted(list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg")))
    if max_images and max_images < len(img_files):
        img_files = img_files[:max_images]

    dataset = []
    total_gt_boxes = 0
    for img_path in img_files:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                    boxes.append((x1, y1, x2, y2))
        dataset.append({"path": img_path, "boxes": boxes})
        total_gt_boxes += len(boxes)

    print(f"Loaded {len(dataset)} HMCHH validation images ({total_gt_boxes} total ground-truth abnormal boxes).")
    return dataset


def evaluate_model_on_hmchh(model_name: str, weights_path: Path, dataset: list, conf: float = 0.111, batch_size: int = 32):
    """Evaluate a model on HMCHH under both unfiltered and abnormal-filtered modes."""
    from ultralytics import YOLO
    config.verify_model_checksum(weights_path)
    model = YOLO(str(weights_path))

    # Determine abnormal class index/indices for this model
    num_classes = len(model.names)
    if num_classes == 5:
        # Phase 1 5-class: Koilocytotic (2) and Dyskeratotic (3) are Abnormal
        abnormal_indices = {2, 3}
    elif num_classes == 2:
        # 2-class: index 1 is Abnormal
        abnormal_indices = {1}
    else:
        abnormal_indices = {1} if 1 in model.names else {0}

    print(f"\nEvaluating {model_name} (Classes: {num_classes}, Abnormal indices: {abnormal_indices})...")
    t0 = time.time()

    total_gt = sum(len(d["boxes"]) for d in dataset)
    unfiltered_preds_total = 0
    filtered_preds_total = 0

    unfiltered_tp_total = 0
    filtered_tp_total = 0

    # Process in batches for high speed
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        img_paths = [str(d["path"]) for d in batch]
        results = model.predict(source=img_paths, conf=conf, imgsz=640, verbose=False, device=0)

        for d, res in zip(batch, results):
            gt_boxes = d["boxes"]
            all_boxes = []
            abn_boxes = []

            if len(res.boxes) > 0:
                boxes_xyxyn = res.boxes.xyxyn.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                for box, cls_id in zip(boxes_xyxyn, classes):
                    b = tuple(box)
                    all_boxes.append(b)
                    if cls_id in abnormal_indices:
                        abn_boxes.append(b)

            unfiltered_preds_total += len(all_boxes)
            filtered_preds_total += len(abn_boxes)

            tp_u, _, _ = match_boxes(all_boxes, gt_boxes)
            tp_f, _, _ = match_boxes(abn_boxes, gt_boxes)

            unfiltered_tp_total += tp_u
            filtered_tp_total += tp_f

    elapsed = time.time() - t0

    # Unfiltered metrics
    u_prec = unfiltered_tp_total / unfiltered_preds_total if unfiltered_preds_total > 0 else 0.0
    u_rec = unfiltered_tp_total / total_gt if total_gt > 0 else 0.0
    u_f1 = 2 * u_prec * u_rec / (u_prec + u_rec) if (u_prec + u_rec) > 0 else 0.0

    # Filtered metrics
    f_prec = filtered_tp_total / filtered_preds_total if filtered_preds_total > 0 else 0.0
    f_rec = filtered_tp_total / total_gt if total_gt > 0 else 0.0
    f_f1 = 2 * f_prec * f_rec / (f_prec + f_rec) if (f_prec + f_rec) > 0 else 0.0

    normal_suppressed = unfiltered_preds_total - filtered_preds_total
    suppression_pct = (normal_suppressed / unfiltered_preds_total * 100) if unfiltered_preds_total > 0 else 0.0

    print(f"Processed {len(dataset)} images in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} FPS)")
    print(f"  Unfiltered:  Preds={unfiltered_preds_total:5d}, TP={unfiltered_tp_total:4d}, Prec={u_prec:.4f}, Rec={u_rec:.4f}, F1={u_f1:.4f}")
    print(f"  Abn-Filter:  Preds={filtered_preds_total:5d}, TP={filtered_tp_total:4d}, Prec={f_prec:.4f}, Rec={f_rec:.4f}, F1={f_f1:.4f}")
    print(f"  Normal false alarms eliminated by protocol alignment: {normal_suppressed} ({suppression_pct:.1f}%)")

    return {
        "model_name": model_name,
        "weights": str(weights_path),
        "total_gt_boxes": total_gt,
        "elapsed_seconds": elapsed,
        "unfiltered": {
            "predicted_boxes": unfiltered_preds_total,
            "true_positives": unfiltered_tp_total,
            "precision": u_prec,
            "recall": u_rec,
            "f1": u_f1,
        },
        "abnormal_filtered": {
            "predicted_boxes": filtered_preds_total,
            "true_positives": filtered_tp_total,
            "precision": f_prec,
            "recall": f_rec,
            "f1": f_f1,
        },
        "suppression_analysis": {
            "normal_detections_suppressed": normal_suppressed,
            "suppression_percentage": suppression_pct,
            "precision_gain_ratio": (f_prec / u_prec) if u_prec > 0 else 0.0,
        },
    }


def main():
    setup_logging("evaluate_hmchh_abnormal_filtered")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmchh-dir", type=Path, default=Path("D:/pap_model/HMCHH_YOLO_prepared"))
    parser.add_argument("--max-images", type=int, default=500, help="Maximum images to evaluate (default 500 for fast high-confidence stats, or 1077 for full split)")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE_CONF)
    args = parser.parse_args()

    if not config.is_path_accessible(args.hmchh_dir):
        print(f"ERROR: HMCHH directory {args.hmchh_dir} is not accessible.", file=sys.stderr)
        sys.exit(1)

    dataset = load_hmchh_val(args.hmchh_dir, max_images=args.max_images)

    models_to_test = [
        ("Exp C2b (Combined 2-class)", config.RESULTS_DIR / "runs" / "expC2b_combined_150" / "weights" / "best.pt"),
        ("Exp C1 (SIPaKMeD 2-class)", config.RESULTS_DIR / "runs" / "expC1_sipakmed_2class" / "weights" / "best.pt"),
        ("Phase 1 (SIPaKMeD 5-class)", config.MODELS_DIR / "best.pt"),
    ]

    report = {
        "dataset_name": "HMCHH-TCT-CellDet (Figshare DOI 10.6084/m9.figshare.27901206)",
        "images_evaluated": len(dataset),
        "conf_threshold": args.conf,
        "models": {},
    }

    print("\n" + "=" * 95)
    print(f"{'Model':<28} | {'Unfilt Prec':<11} | {'Filt Prec':<10} | {'Prec Gain':<10} | {'Abn Recall':<10} | {'F1':<8}")
    print("=" * 95)

    for name, path in models_to_test:
        if path.exists():
            res = evaluate_model_on_hmchh(name, path, dataset, conf=args.conf)
            report["models"][name] = res
            u_p = res["unfiltered"]["precision"]
            f_p = res["abnormal_filtered"]["precision"]
            gain = res["suppression_analysis"]["precision_gain_ratio"]
            f_r = res["abnormal_filtered"]["recall"]
            f_f1 = res["abnormal_filtered"]["f1"]
            print(f"{name:<28} | {u_p:<11.4f} | {f_p:<10.4f} | {gain:<9.2f}x | {f_r:<10.4f} | {f_f1:<8.4f}")

    print("=" * 95)
    out_json = config.RESULTS_DIR / "hmchh_abnormal_filtered_evaluation.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved HMCHH evaluation report to: {out_json}")


if __name__ == "__main__":
    main()
