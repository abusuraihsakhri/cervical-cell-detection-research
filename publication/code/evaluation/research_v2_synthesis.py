"""Paired field uncertainty, seed variation, and reference-sensitive stain analysis."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.research_v2 import ROOT, OUT, collect, load_folder, field_counts, ratios, summarize, write_json
from evaluation.stain_normalization_benchmark import ReinhardNormalizer


def paired_difference(first, second, iterations=1000):
    if set(first) != set(second):
        raise ValueError('Paired comparison requires identical fields')
    groups = sorted({v['group'] for v in first.values()})
    a, b = [], []
    for group in groups:
        names = [n for n in first if first[n]['group'] == group]
        if any(second[n]['group'] != group for n in names):
            raise ValueError('Group mismatch')
        a.append(np.sum([first[n]['counts'] for n in names], axis=0))
        b.append(np.sum([second[n]['counts'] for n in names], axis=0))
    a, b = np.array(a), np.array(b)
    ma, mb = ratios(a.sum(axis=0)), ratios(b.sum(axis=0))
    rng = np.random.default_rng(20260921)
    samples = []
    for _ in range(iterations):
        indexes = rng.integers(0, len(groups), len(groups))
        x, y = ratios(a[indexes].sum(axis=0)), ratios(b[indexes].sum(axis=0))
        samples.append({key:y[key]-x[key] for key in x})
    return {'direction': 'second minus first', 'groups': len(groups), 'delta': {k:mb[k]-ma[k] for k in ma},
            'paired_cluster_bootstrap_95ci': {k:np.quantile([s[k] for s in samples], [.025,.975]).tolist() for k in ma},
            'limitation': 'Conditional on fixed trained models and development-selected thresholds; not a prospective test or multiple-comparison-adjusted inference'}


def main():
    report = json.loads((OUT/'evaluation_summary.json').read_text())
    models, protocol = report['models'], report['protocol']
    paired, variability = {}, {}
    for seed in (17,43,101):
        single, combined = (models[f'v2_{v}_seed{seed}'] for v in ('sipakmed_only', 'combined'))
        paired[str(seed)] = {endpoint: paired_difference(single[field], combined[field]) for endpoint,field in [('dense','dense_per_field'),('hmchh','hmchh_per_field')]}
    for variant in ('sipakmed_only','combined'):
        variability[variant] = {}
        for endpoint in ('dense_evaluation','hmchh'):
            values = [models[f'v2_{variant}_seed{s}'][endpoint]['metrics'] for s in (17,43,101)]
            variability[variant][endpoint] = {k:{'mean':float(np.mean([v[k] for v in values])), 'min':min(v[k] for v in values),'max':max(v[k] for v in values),'sample_sd':float(np.std([v[k] for v in values],ddof=1))} for k in values[0]}
    write_json(OUT/'paired_seed_comparisons.json', {'paired_by_seed':paired,'seed_variability':variability,'note':'Three explicitly different seeds. No universal noise floor is inferred.'})
    dense = load_folder(ROOT/'data/dense_eval/sipakmed_dense40_verified')
    dense_test = {n:dense[n] for n in protocol['dense_evaluation']}
    sparse = load_folder(ROOT/'data/research_v2/combined/test')
    apc = {n:v for n,v in sparse.items() if n.startswith('apcdata__')}
    analysis_protocol = json.loads((OUT/'analysis_protocol.json').read_text())
    reference_sets = [[Path(p) for p in row] for row in analysis_protocol['reference_sets']]
    from evaluation.research_v2 import digest
    for images in reference_sets:
        for path in images:
            if digest(path) != analysis_protocol['reference_hashes'][str(path)]:
                raise ValueError('Frozen stain reference changed')
    stain = {'method':'Reinhard CIELAB inference-time transfer','reference_sets':[[p.name for p in ps] for ps in reference_sets],
             'scope':'Three reference sets and three model seeds. Paired descriptive uncertainty; no morphology-invariance or universal-normalization claim. APCData sparse labels limit precision interpretation.','models':{}}
    for name, model_report in models.items():
        weights = OUT/'runs'/name/'weights/best.pt'
        threshold = model_report['threshold_from_development']
        stain['models'][name] = {}
        for dataset_name, data in [('dense_evaluation',dense_test),('apc_sparse_test',apc)]:
            raw = collect(weights,data,protocol['inference'],name+'_'+dataset_name+'_stainraw')
            raw_counts = field_counts(data,raw,threshold)
            rows=[]
            for i, images in enumerate(reference_sets):
                normalizer=ReinhardNormalizer()
                normalizer.fit(images)
                normalized=collect(weights,data,protocol['inference'],name+'_'+dataset_name+f'_ref{i}',normalizer)
                normalized_counts=field_counts(data,normalized,threshold)
                rows.append({'reference_index':i,'raw':summarize(raw_counts),'normalized':summarize(normalized_counts),
                             'paired_difference':paired_difference(raw_counts,normalized_counts)})
            stain['models'][name][dataset_name]=rows
        write_json(OUT/'stain_reference_sensitivity.json',stain)
        print(f'Completed stain reference analysis: {name}',flush=True)


if __name__ == '__main__':
    main()
