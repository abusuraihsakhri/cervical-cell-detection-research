# Publication Package

Manuscript, supplementary materials, figures, tables, certified trained weights, source code snapshot, and verification evidence for:

**Hidden field overlap, incomplete reference annotation and label conversion errors in cervical-cell detection: a retrospective multi-source study with external evaluation**

Everything in this release package was generated from the frozen v3 protocol. Figures and tables are generated via `scripts/build_figures_tables.py`, and compiled Word files are produced via `scripts/build_docx.py`.

---

## Contents

| Path | Description |
|---|---|
| `manuscript/manuscript.md`, `.docx` | Complete manuscript with citations and declarations |
| `manuscript/references_verification.md` | Verification register for every cited publication (registry and accession date) |
| `supplementary/supplementary_material.md`, `.docx` | Supplementary materials: S1 prior analysis corrections, S2 overlap detection, S3 dense annotation provenance, S4 per-seed metrics, S5 reporting checklist |
| `figures/` | Figures 1–4 and S1 in high-resolution PNG (300 dpi) and vector PDF |
| `tables/` | Numerical results tables in CSV format (Tables 1–3, Tables S2–S7) |
| `weights/` | Six trained YOLOv8n checkpoints (`best.pt`), training arguments (`args.yaml`), per-epoch results, and `MODEL_CARD.md` |
| `code/` | Frozen snapshot of data preparation, training, evaluation, and visualization scripts |
| `reproducibility/` | Frozen protocols, data manifest, compressed overlap evidence (`field_overlap.json.gz`), release verification report, and SHA-256 checksums |
| `results/` | Machine-readable JSON results, including per-field and bootstrap confidence intervals |

---

## Reproducing the Study

Refer to `code/REPRODUCE.md` for complete reproduction instructions and environment details.
All artifacts can be verified locally using the automated verification suite:

```bash
python publication/scripts/verify_release.py
```
