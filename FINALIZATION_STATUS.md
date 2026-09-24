# Research finalization status

Status on 19 September 2026: audit corrections complete; study rerun as version 3; publication package assembled in `publication/`. Remaining items need the authors (listed at the end).

| Requirement | Evidence | Status |
|---|---|---|
| Correct class mapping | Name-based taxonomy; regression tests; baseline rerun reproduces 166/361 abnormal cells | Done |
| Reject unreviewed annotations | Fail-closed Label Studio importer; tests | Done |
| Annotation provenance | User attestation and export hash recorded; single reviewer; no second review | Documented as a limitation |
| Independent splits | Exhaustive within-source overlap matching (1,019,514 pairs; 791 overlaps, all pixel-confirmed); component-grouped splits; 0 crossings | Done (v3). v2 was found to leak and was superseded |
| External reference | HMCHH rebuilt from XML: abnormal cells only (1,837), organisms excluded, polygons included | Done |
| Controlled repeated training | Six runs, seeds 17/43/101, identical budget and optimizer | Done |
| Independent threshold selection | Component-level 20/20 dense development/evaluation split, frozen before training | Done |
| Calibration | Exact NLL fit on development, scored on evaluation | Done |
| Uncertainty and paired comparisons | Cluster bootstrap; paired seed comparisons | Done |
| Stain analysis | Three reference sets, paired intervals, descriptive claims only | Done |
| Reproducibility | Frozen protocols, manifests, environment, checksums; release verification 29/29 checks; 16 tests pass | Done |
| Manuscript and supplement | `publication/manuscript/`, `publication/supplementary/`; 29 references verified against registries | Done, pending author details |

## Items only the authors can complete

- Author list, affiliations, corresponding author, funding, competing interests, CRediT statement, AI-tool declaration and repository URL (marked ⟦…⟧ in the manuscript and cover letter).
- Confirm the ethics statement with the institution.
- Ideally, an independent second review of the 40 dense fields by a qualified cytologist before submission; the manuscript currently reports this as a limitation.
- Check dataset licences (SIPaKMeD derivative, HMCHH) before publishing the dense-reference label files.
