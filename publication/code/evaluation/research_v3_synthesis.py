"""Paired seed comparisons, seed variability and stain reference sensitivity (version 3)."""
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.research_v2 import collect, field_counts, summarize, write_json, digest
from evaluation.research_v2_synthesis import paired_difference
from evaluation.research_v3 import OUT, CACHE, SEEDS, VARIANTS, load_dense, load_sparse_test
from evaluation.stain_normalization_benchmark import ReinhardNormalizer


def main():
    report = json.loads((OUT / 'evaluation_summary.json').read_text())
    models, protocol = report['models'], report['protocol']
    paired, variability = {}, {}
    for seed in SEEDS:
        single, combined = (models[f'v3_{v}_seed{seed}'] for v in VARIANTS)
        paired[str(seed)] = {endpoint: paired_difference(single[field], combined[field])
                             for endpoint, field in (('dense', 'dense_per_field'), ('hmchh', 'hmchh_per_field'))}
    for variant in VARIANTS:
        variability[variant] = {}
        for endpoint in ('dense_evaluation', 'hmchh'):
            values = [models[f'v3_{variant}_seed{s}'][endpoint]['metrics'] for s in SEEDS]
            variability[variant][endpoint] = {k: {'mean': float(np.mean([v[k] for v in values])),
                                                  'min': min(v[k] for v in values), 'max': max(v[k] for v in values),
                                                  'sample_sd': float(np.std([v[k] for v in values], ddof=1))} for k in values[0]}
    write_json(OUT / 'paired_seed_comparisons.json', {'paired_by_seed': paired, 'seed_variability': variability,
                                                     'note': 'Three explicitly different seeds. No universal noise floor is inferred.'})
    dense = load_dense(protocol)
    dense_test = {n: dense[n] for n in protocol['dense_evaluation']}
    apc = {n: v for n, v in load_sparse_test().items() if n.startswith('apcdata__')}
    analysis = json.loads((OUT / 'analysis_protocol.json').read_text())
    reference_sets = [[Path(p) for p in row] for row in analysis['reference_sets']]
    for images in reference_sets:
        for path in images:
            if digest(path) != analysis['reference_hashes'][str(path)]:
                raise ValueError('Frozen stain reference changed')
    stain = {'method': 'Reinhard CIELAB inference-time transfer', 'reference_sets': [[p.name for p in s] for s in reference_sets],
             'scope': analysis['claims'], 'models': {}}
    for name, model_report in models.items():
        weights = OUT / 'runs' / name / 'weights/best.pt'
        threshold = model_report['threshold_from_development']
        stain['models'][name] = {}
        for dataset_name, data in (('dense_evaluation', dense_test), ('apc_sparse_test', apc)):
            raw = collect(weights, data, protocol['inference'], f'{name}_{dataset_name}_stainraw', cache_dir=CACHE)
            raw_counts = field_counts(data, raw, threshold)
            rows = []
            for i, images in enumerate(reference_sets):
                normalizer = ReinhardNormalizer()
                normalizer.fit(images)
                normalized = collect(weights, data, protocol['inference'], f'{name}_{dataset_name}_ref{i}', normalizer, cache_dir=CACHE)
                normalized_counts = field_counts(data, normalized, threshold)
                rows.append({'reference_index': i, 'raw': summarize(raw_counts), 'normalized': summarize(normalized_counts),
                             'paired_difference': paired_difference(raw_counts, normalized_counts)})
            stain['models'][name][dataset_name] = rows
        write_json(OUT / 'stain_reference_sensitivity.json', stain)
        print(f'Completed stain reference analysis: {name}', flush=True)


if __name__ == '__main__':
    main()
