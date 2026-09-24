#!/bin/bash
# Full v3 pipeline after the split and protocol are frozen:
#   train six models -> evaluate -> synthesis -> secondary analysis -> figures/tables -> release verification.
# Stops at the first failing stage.
set -e
cd "$(dirname "$0")/../.."
python -u run_research_v3.py                                   > results/research_v3/training_runner.log 2>&1
python -u evaluation/research_v3.py                            > results/research_v3/evaluation.log 2>&1
python -u evaluation/research_v3_synthesis.py                  > results/research_v3/synthesis.log 2>&1
python -u publication/scripts/annotation_completeness_analysis.py > results/research_v3/completeness.log 2>&1
python -u publication/scripts/conversion_effect_analysis.py     > results/research_v3/conversion_effect.log 2>&1
python -u publication/scripts/build_figures_tables.py          > results/research_v3/figures.log 2>&1
PYTHONIOENCODING=utf-8 python -u publication/scripts/verify_release.py > results/research_v3/verification.log 2>&1
echo PIPELINE_DONE
