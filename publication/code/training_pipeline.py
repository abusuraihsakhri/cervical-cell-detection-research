"""
YOLOv8n training pipeline — Section 2 & Section 5 of AGENT-SPEC-FINAL.md.

Epoch strategy is a STOPPING RULE, not a fixed target: --epochs is a
ceiling/safety cap, --patience triggers early stopping on the installed Ultralytics fitness criterion
plateau. Do not trust any timing/convergence-epoch guess in the spec —
log real per-epoch wall-clock time from the first few epochs instead.

Usage:
    python training_pipeline.py --data data/sipakmed/primary/data.yaml
    python training_pipeline.py --data ... --model yolov8s.pt   # escalation, only if justified
"""

import argparse
import json
import time
from pathlib import Path

import config
from _logging_setup import setup_logging


def _oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg 


def train_with_batch_fallback(model_name: str, data_yaml: str, base_kwargs: dict):
    """
    Try training at the configured batch size, dropping down the
    BATCH_FALLBACK_SEQUENCE on CUDA OOM, per spec Section 2:
    "drop to 8, then 4, if OOM".
    """
    from ultralytics import YOLO

    # Verify cryptographic integrity of model weights before execution (OWASP A08:2025)
    config.verify_model_checksum(model_name, enforce=True)

    last_exc = None
    for batch in config.BATCH_FALLBACK_SEQUENCE:
        print(f"--- Attempting training at batch={batch} ---")
        model = YOLO(model_name)
        kwargs = {**base_kwargs, "batch": batch}
        t0 = time.time()
        try:
            results = model.train(data=data_yaml, **kwargs)
            elapsed = time.time() - t0
            print(f"Training succeeded at batch={batch} in {elapsed:.1f}s total.")
            return results, batch, elapsed, Path(model.trainer.save_dir)
        except Exception as e:  # noqa: BLE001
            if _oom_error(e):
                print(f"OOM at batch={batch}, falling back to next smaller batch size.")
                last_exc = e
                del model
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(
        f"Training failed at every batch size in {config.BATCH_FALLBACK_SEQUENCE}. "
        f"Last error: {last_exc}"
    )


def log_first_epochs_timing(results_dir: Path) -> dict:
    """
    Parse the ultralytics results.csv (if present) to report real
    per-epoch wall-clock time from the first few epochs, per spec
    Section 5's instruction to correct the guessed timing estimate
    with actual hardware numbers.
    """
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return {}
    import csv as csv_mod

    rows = list(csv_mod.DictReader(csv_path.open()))
    if len(rows) < 2:
        return {}
    # ultralytics results.csv doesn't include wall time by default in all
    # versions; if unavailable, this just reports epoch count reached.
    return {
        "epochs_run": len(rows),
        "note": (
            "Compare epochs_run against --patience/--epochs ceiling to see "
            "whether training plateaued early or hit the cap. Check the "
            "training log's printed per-epoch time directly for wall-clock "
            "figures; results.csv format varies by ultralytics version."
        ),
    }


def main():
    setup_logging("training_pipeline")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to data.yaml (YOLOv8 format)")
    parser.add_argument(
        "--model",
        default=config.PRIMARY_MODEL,
        help=f"Model checkpoint to start from (default: {config.PRIMARY_MODEL}). "
        f"Do not exceed {config.MAX_ALLOWED_MODEL} without an explicit documented decision.",
    )
    parser.add_argument("--project", default=str(config.RESULTS_DIR / "runs"))
    parser.add_argument("--name", default="phase1_sipakmed")
    parser.add_argument(
        "--epochs", type=int, default=config.TRAINING_CONFIG["epochs"],
        help="Ceiling, not a target. The Phase 1 run plateaued at epoch ~50-60 and "
             "early-stopped at 101, so 100 is ample for this data at 640px; raise it "
             "for heavily augmented runs, which converge later.",
    )
    parser.add_argument(
        "--patience", type=int, default=config.TRAINING_CONFIG["patience"],
        help="Early-stopping patience on the installed Ultralytics fitness criterion.",
    )
    parser.add_argument(
        "--workers", type=int, default=config.TRAINING_CONFIG["workers"],
        help="Dataloader workers. Lower this on a memory-constrained machine — a "
             "combined-dataset run was killed by the OS low-memory reaper at 4 workers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model not in (config.PRIMARY_MODEL, config.FALLBACK_MODEL):
        print(
            f"WARNING: model '{args.model}' is not the spec-sanctioned primary "
            f"({config.PRIMARY_MODEL}) or fallback ({config.FALLBACK_MODEL}). "
            "Per Section 8 non-goals, no model larger than YOLOv8s without an "
            "explicit decision to accept slower training / smaller batch."
        )

    base_kwargs = {
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": config.TRAINING_CONFIG["imgsz"],
        "amp": True,
        "seed": args.seed,
        "deterministic": True,
        "cache": config.TRAINING_CONFIG["cache"],
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "pretrained": True,  # supplied pretrained YOLO detection checkpoint
    }

    results, used_batch, elapsed, run_dir = train_with_batch_fallback(args.model, args.data, base_kwargs)

    timing = log_first_epochs_timing(run_dir)

    summary = {
        "model": args.model,
        "seed": args.seed,
        "save_dir": str(run_dir.resolve()),
        "initial_weights_sha256": config.compute_file_sha256(args.model),
        "data": args.data,
        "batch_used": used_batch,
        "workers": args.workers,
        "total_wall_clock_seconds": elapsed,
        "epochs_ceiling": args.epochs,
        "patience": args.patience,
        **timing,
    }
    summary_path = config.RESULTS_DIR / f"{args.name}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nTraining summary written to {summary_path}")
    print(json.dumps(summary, indent=2))
    print(
        "\nBest checkpoint should be at: "
        f"{run_dir / 'weights' / 'best.pt'}\n"
        "Copy it into models/ (gitignored except best.pt, per repo structure) "
        "before running evaluation scripts."
    )


if __name__ == "__main__":
    main()
