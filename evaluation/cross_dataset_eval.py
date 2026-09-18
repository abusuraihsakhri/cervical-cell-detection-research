"""
Section 6b — Cross-dataset generalization (Phase 2).

- Train on SIPaKMeD only (handled by training_pipeline.py).
- Zero-shot evaluate on CRIC (or Mendeley LBC fallback) with NO fine-tuning.
- Report the accuracy/mAP/recall delta vs. Phase 1 SIPaKMeD validation results.
- Explicitly compare against the Coskun et al. 2026 benchmark (91% accuracy /
  0.91 macro-F1), noting that benchmark used proprietary multi-center WSI
  data this project does not have. A lower number here is the expected
  "public-data-only, YOLO-detection ceiling," not a failure.
- If second-dataset access fails entirely, this script documents that as a
  stated limitation rather than omitting the section.

Usage (defaults now point at the real prepared data.yaml files, so this
can be run with just --weights):
    python evaluation/cross_dataset_eval.py --weights models/best.pt
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from _logging_setup import setup_logging
from confusion_matrix import build_confusion_matrix, per_class_precision_recall, class_agnostic_detection_metrics


def macro_f1(metrics: dict) -> float:
    f1s = [m["f1"] for m in metrics.values()]
    return sum(f1s) / len(f1s) if f1s else 0.0


def overall_accuracy_from_cm(cm) -> float:
    import numpy as np

    arr = np.array(cm)
    correct = np.trace(arr[: arr.shape[0] - 1, : arr.shape[1] - 1])  # exclude background row/col
    total = arr[: arr.shape[0] - 1, : arr.shape[1] - 1].sum()
    return float(correct / total) if total > 0 else 0.0


def main():
    setup_logging("cross_dataset_eval")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--phase1-data", default=str(config.SIPAKMED_DATA_YAML),
        help="SIPaKMeD data.yaml (in-distribution reference)",
    )
    parser.add_argument(
        "--phase2-data", default=str(config.APCDATA_DATA_YAML),
        help="APCData data.yaml (zero-shot target) — see config.PHASE2_DATASET",
    )
    parser.add_argument(
        "--phase2-dataset-name",
        default=config.PHASE2_DATASET_SOURCES[config.PHASE2_DATASET]["name"],
    )
    parser.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE_CONF)
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Defaults to a per-dataset filename derived from "
             "--phase2-dataset-name, so evaluating a second target dataset does not "
             "overwrite the first one's results (it did once — the APCData run was "
             "lost to the HMCHH-TCT run and had to be recovered from the log).",
    )
    args = parser.parse_args()

    report = {"benchmark_comparison": config.COSKUN_2026_BENCHMARK}

    print("Evaluating in-distribution (SIPaKMeD) performance for delta baseline...")
    cm1, names1 = build_confusion_matrix(args.weights, args.phase1_data, args.conf)
    metrics1 = per_class_precision_recall(cm1, names1)
    acc1 = overall_accuracy_from_cm(cm1)
    f1_1 = macro_f1(metrics1)
    report["phase1_sipakmed_indistribution"] = {"accuracy": acc1, "macro_f1": f1_1, "per_class": metrics1}

    if args.phase2_data is None or not config.is_path_accessible(Path(args.phase2_data)):
        report["phase2_cross_dataset"] = {
            "status": "NOT RUN — second dataset unavailable",
            "limitation_statement": (
                f"Cross-dataset generalization on '{args.phase2_dataset_name}' could not be "
                "evaluated because the dataset was not accessible or its data.yaml was not "
                "found. Per spec Section 6b, this is documented explicitly as a stated "
                "limitation rather than omitted from the report."
            ),
        }
        print("WARNING: Phase 2 dataset unavailable — recording as a stated limitation, not omitting the section.")
    else:
        print(f"Zero-shot evaluating on {args.phase2_dataset_name} (no fine-tuning)...")
        cm2, names2 = build_confusion_matrix(args.weights, args.phase2_data, args.conf)
        metrics2 = per_class_precision_recall(cm2, names2)
        acc2 = overall_accuracy_from_cm(cm2)
        f1_2 = macro_f1(metrics2)

        detection_only2 = class_agnostic_detection_metrics(cm2)
        detection_only1 = class_agnostic_detection_metrics(cm1)

        report["phase2_cross_dataset"] = {
            "dataset_name": args.phase2_dataset_name,
            "accuracy": acc2,
            "macro_f1": f1_2,
            "per_class": metrics2,
            "delta_vs_phase1": {"accuracy_delta": acc2 - acc1, "macro_f1_delta": f1_2 - f1_1},
            "delta_vs_coskun_2026_benchmark": {
                "accuracy_delta": acc2 - config.COSKUN_2026_BENCHMARK["accuracy"],
                "macro_f1_delta": f1_2 - config.COSKUN_2026_BENCHMARK["macro_f1"],
            },
            "framing_note": (
                "The Coskun et al. 2026 benchmark used proprietary multi-center WSI data "
                "in addition to public sources. A lower number here should be discussed as "
                "the public-data-only, YOLO-detection ceiling, not treated as a failure "
                "(spec Section 0 / 6b)."
            ),
            "class_agnostic_detection": {
                "phase1_sipakmed": detection_only1,
                "phase2_target": detection_only2,
                "caveat": (
                    f"SIPaKMeD's classes (cell morphology) and {args.phase2_dataset_name}'s "
                    "classes are a different taxonomy (or, for a binary dataset, a different "
                    "granularity) with no defensible 1:1 mapping to SIPaKMeD's classes, so "
                    "the accuracy/macro_f1 numbers above conflate 'wrong class' with "
                    "'meaningless class comparison'. These class-agnostic precision/recall/F1 "
                    "numbers instead measure only whether the detector localizes a cell-like "
                    "object at all (IoU>=0.5), independent of class identity, isolating true "
                    "cross-dataset domain shift from the taxonomy mismatch."
                ),
            },
        }

    if args.out:
        out_path = Path(args.out)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", args.phase2_dataset_name.lower()).strip("_")
        out_path = config.RESULTS_DIR / f"cross_dataset_eval_{slug}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCross-dataset report written to {out_path}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
