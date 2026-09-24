# Publication package

Manuscript, supplement, figures, tables, trained weights, code snapshot and verification evidence for:

**Hidden field overlap, incomplete reference annotation and label conversion errors in cervical-cell detection: a retrospective multi-source study with external evaluation**

Everything here was generated from the frozen v3 protocol. Nothing in this folder is hand-edited output: figures and tables come from `scripts/build_figures_tables.py`, and the Word files come from `scripts/build_docx.py`.

## Contents

| Path | What it is |
|---|---|
| `manuscript/manuscript.md`, `.docx` | Main text with references |
| `manuscript/cover_letter.md`, `.docx` | Cover letter draft |
| `manuscript/references_verification.md` | How each reference was verified (registry and date) |
| `supplementary/supplementary_material.md`, `.docx` | S1 corrections to earlier analyses, S2 overlap detection, S3 dense annotation provenance, S4 per-seed results, S5 reporting checklist |
| `figures/` | Figures 1–4 and S1, PNG (300 dpi) and PDF |
| `tables/` | All tables as CSV |
| `weights/` | Six trained checkpoints (`best.pt`), training arguments, per-epoch logs, and `MODEL_CARD.md` |
| `code/` | Snapshot of the code used for data preparation, training, evaluation and figures |
| `reproducibility/` | Frozen protocols, data manifest, overlap evidence, environment, release verification report, SHA-256 checksums |
| `results/` | Machine-readable results, including per-field counts |

## Placeholders the authors must complete

Author names, affiliations, corresponding author, funding, competing interests, CRediT contributions, the ethics statement confirmation, the AI-tool declaration and the repository URL are marked ⟦…⟧ in the manuscript. Nothing else is left as a placeholder.

## Reproducing the study

See `code/REPRODUCE.md`.
