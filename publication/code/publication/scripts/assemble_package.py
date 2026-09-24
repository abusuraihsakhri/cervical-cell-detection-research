"""Assemble the publication package: code snapshot, weights, results, evidence, checksums.

Dataset images are never copied. The dense reference is released as YOLO label
files keyed by filename and SHA-256 (the Label Studio export embeds images).

    python publication/scripts/assemble_package.py
"""
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / 'publication'
V3 = ROOT / 'results/research_v3'
SEEDS, VARIANTS = (17, 43, 101), ('sipakmed_only', 'combined')

CODE = ['config.py', '_logging_setup.py', 'training_pipeline.py', 'prepare_apcdata.py', 'prepare_hmchh.py',
        'prepare_combined_2class.py', 'prepare_research_v2.py', 'prepare_research_v3.py',
        'prepare_hmchh_abnormal_reference.py', 'run_research_v3.py', 'requirements.txt',
        'evaluation/__init__.py', 'evaluation/taxonomy.py', 'evaluation/calibration.py',
        'evaluation/sweep_operating_thresholds.py', 'evaluation/stain_normalization_benchmark.py',
        'evaluation/research_v2.py', 'evaluation/research_v2_synthesis.py', 'evaluation/research_v3.py',
        'evaluation/research_v3_synthesis.py', 'evaluation/confusion_matrix.py',
        'annotation/__init__.py', 'annotation/evaluate_dense.py', 'annotation/pre_annotate.py',
        'tests/test_evaluation_integrity.py', 'tests/test_research_v3.py',
        'publication/__init__.py', 'publication/scripts/__init__.py',
        'publication/scripts/detect_field_overlap.py', 'publication/scripts/verify_overlap_photometric.py',
        'publication/scripts/verify_release.py',
        'publication/scripts/annotation_completeness_analysis.py', 'publication/scripts/conversion_effect_analysis.py', 'publication/scripts/build_figures_tables.py',
        'publication/scripts/build_docx.py', 'publication/scripts/assemble_package.py',
        'publication/scripts/run_pipeline_v3.sh']
EVIDENCE = ['evaluation_protocol.json', 'analysis_protocol.json', 'protocol.json', 'data_manifest.json',
            'dense_split.json', 'field_overlap.json.gz', 'overlap_photometric.json', 'hmchh_reference_summary.json',
            'training_status.json']
RESULTS = ['evaluation_summary.json', 'paired_seed_comparisons.json', 'stain_reference_sensitivity.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    for rel in CODE:
        copy(ROOT / rel, PUB / 'code' / rel)
    for name in EVIDENCE:
        copy(V3 / name, PUB / 'reproducibility' / name)
    copy(ROOT / 'results/research_v2/training_fitness_source.txt', PUB / 'reproducibility/training_fitness_source.txt')
    copy(ROOT / 'results/research_v2/environment_snapshot.json', PUB / 'reproducibility/environment_snapshot.json')
    copy(ROOT / 'results/annotation_attestation_20260919.json', PUB / 'reproducibility/annotation_attestation_20260919.json')
    # overlap_calibration contains exploratory screening matrix (>76MB); not needed for publication release
    for name in RESULTS:
        copy(V3 / name, PUB / 'results' / name)
    for seed in SEEDS:
        for variant in VARIANTS:
            name = f'v3_{variant}_seed{seed}'
            run = V3 / 'runs' / name
            for rel in ('weights/best.pt', 'args.yaml', 'results.csv'):
                copy(run / rel, PUB / 'weights' / name / rel)
            copy(ROOT / 'results' / f'{name}_training_summary.json', PUB / 'weights' / name / 'training_summary.json')
            copy(V3 / f'{name}_evaluation.json', PUB / 'results' / f'{name}_evaluation.json')
    labels = ROOT / 'data/dense_eval/sipakmed_dense40_verified/labels'
    for label in labels.glob('*.txt'):
        copy(label, PUB / 'reproducibility/dense_reference_labels' / label.name)
    files = [p for p in sorted(PUB.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.txt'
             and '__pycache__' not in p.parts and 'manuscript' not in p.parts and 'supplementary' not in p.parts]
    names = [str(p.relative_to(PUB)).replace('\\', '/') for p in files]
    images = [n for n in names if n.lower().endswith(('.jpg', '.jpeg', '.bmp')) or
              (n.lower().endswith('.png') and not n.startswith('figures/'))]
    if images:
        raise RuntimeError(f'Dataset images must not be packaged: {images[:5]}')
    checksums = {n: sha(p) for n, p in zip(names, files)}
    (PUB / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n, h in checksums.items()), encoding='utf-8')
    print(f'{len(checksums)} files; checksums written to SHA256SUMS.txt')


if __name__ == '__main__':
    main()
