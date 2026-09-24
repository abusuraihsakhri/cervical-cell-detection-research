"""Build manuscript figures (PNG + PDF) and tables (CSV) from frozen result files.

Reads only saved outputs; performs no inference.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / 'results/research_v3'
PUB = ROOT / 'publication'
FIG, TAB = PUB / 'figures', PUB / 'tables'
SEEDS = (17, 43, 101)
VARIANTS = {'sipakmed_only': 'SIPaKMeD only', 'combined': 'SIPaKMeD + APCData'}
COLORS = {'sipakmed_only': '#2a78d6', 'combined': '#eb6834'}
MARKERS = {'sipakmed_only': 'o', 'combined': 's'}
N_EVAL = None
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.dpi': 300, 'figure.facecolor': 'white',
})


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def write_csv(name, header, rows):
    with open(TAB / f'{name}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fmt(value, ci=None, pct=True):
    scale = 100 if pct else 1
    text = f'{value*scale:.1f}' if pct else f'{value:.2f}'
    if ci is not None:
        lo, hi = (f'{c*scale:.1f}' if pct else f'{c:.2f}' for c in ci)
        sep = ' to ' if min(ci) < 0 else '–'
        text += f' ({lo}{sep}{hi})'
    return text.replace('-', '−')


def model_name(variant, seed):
    return f'v3_{variant}_seed{seed}'


def figure_study_design(manifest, protocol):
    c = manifest['counts']
    n_dev, n_eval = len(protocol['dense_development']), len(protocol['dense_evaluation'])
    ext = protocol['external']
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.3)

    def box(x, y, w, h, text, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    fc='#f4f3f0', ec=MUTED, lw=0.8))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=6.8, color=INK, weight=weight,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))

    box(0.1, 4.4, 3.0, 1.8, 'SIPaKMeD (Roboflow derivative)\ngrouped by field name\n+ image overlap\n'
        f'{manifest["quarantine_counts"]["sipakmed"]} fields excluded\n(unresolved provenance)')
    box(3.5, 4.4, 3.0, 1.8, 'APCData (Mendeley)\ngrouped by specimen code\n+ image overlap\n'
        f'{manifest["quarantine_counts"]["apcdata"]} fields excluded')
    box(6.9, 4.4, 3.0, 1.8, 'HMCHH-TCT-CellDet\nvalidation split,\nabnormal cells only;\nnever used for training\nor tuning')
    box(0.1, 2.6, 6.4, 1.4, 'Component-disjoint splits (15% test, 15% validation)\n'
        f'SIPaKMeD train/valid/test {c["sipakmed/train"]}/{c["sipakmed/valid"]}/{c["sipakmed/test"]} · '
        f'APCData {c["apcdata/train"]}/{c["apcdata/valid"]}/{c["apcdata/test"]}\n'
        'YOLOv8n, 2 training sets × 3 seeds (17, 43, 101), identical budget')
    box(0.1, 0.3, 3.0, 1.9, 'Dense reference:\n40 SIPaKMeD fields\n(1,067 cells; single reviewer)\n'
        'overlap components withheld\nfrom all training')
    box(3.5, 0.3, 3.0, 1.9, f'Frozen component-level split\n{n_dev} development fields\n→ threshold, temperature\n'
        f'{n_eval} evaluation fields\n→ reported once')
    box(6.9, 0.3, 3.0, 3.7, f'External evaluation\n{ext["fields"]:,} HMCHH fields\n{ext["reference_cells"]:,} abnormal cells\n\n'
        'threshold carried over\nfrom dense development;\nno retuning')
    arrow(1.6, 4.4, 1.6, 4.0)
    arrow(5.0, 4.4, 5.0, 4.0)
    arrow(8.4, 4.4, 8.4, 4.0)
    arrow(1.6, 2.6, 1.6, 2.2)
    arrow(3.1, 1.25, 3.5, 1.25)
    arrow(6.5, 1.25, 6.9, 1.25)
    save(fig, 'figure1_study_design')


def dot_panel(ax, models, endpoint, metric, label):
    offsets = {'sipakmed_only': -0.12, 'combined': 0.12}
    for variant in VARIANTS:
        for i, seed in enumerate(SEEDS):
            r = models[model_name(variant, seed)][endpoint]
            value = r['metrics'][metric]
            lo, hi = r['conditional_cluster_bootstrap_95ci'][metric]
            x = i + offsets[variant]
            ax.plot([x, x], [lo, hi], color=COLORS[variant], lw=2, solid_capstyle='round')
            ax.plot(x, value, MARKERS[variant], color=COLORS[variant], ms=6, mec='white', mew=1,
                    label=VARIANTS[variant] if i == 0 else None)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f'seed {s}' for s in SEEDS])
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
    ax.grid(axis='x', visible=False)


def figure_results(models):
    panels = [('dense_evaluation', 'abnormal_recall', 'Abnormal-cell recall'),
              ('dense_evaluation', 'abnormal_precision', 'Abnormal-cell precision'),
              ('hmchh', 'abnormal_recall', 'Abnormal-cell recall'),
              ('hmchh', 'abnormal_precision', 'Abnormal-cell precision')]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, (endpoint, metric, label), tag in zip(axes.flat, panels, 'abcd'):
        dot_panel(ax, models, endpoint, metric, label)
        where = f'Dense SIPaKMeD evaluation ({N_EVAL} fields)' if endpoint == 'dense_evaluation' else 'External HMCHH (1,077 fields)'
        ax.set_title(f'({tag}) {where}', fontsize=8.5, loc='left', color=INK)
    axes[0, 0].legend(loc='lower left', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'figure2_primary_and_external')


def figure_annotation(completeness):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, metric, title in zip(axes, ('abnormal_precision', 'abnormal_recall'),
                                 ('(a) Abnormal-cell precision', '(b) Abnormal-cell recall')):
        for v_index, variant in enumerate(VARIANTS):
            for s_index, seed in enumerate(SEEDS):
                r = completeness['models'][model_name(variant, seed)]
                sparse = r['sparse_reference']['metrics'][metric]
                dense = r['dense_reference']['metrics'][metric]
                y = v_index * 4 + s_index
                ax.plot([sparse, dense], [y, y], color=GRID, lw=2, zorder=1)
                ax.plot(sparse, y, MARKERS[variant], mfc='white', mec=COLORS[variant], ms=6, zorder=2,
                        label='Sparse reference' if v_index == 0 and s_index == 0 else None)
                ax.plot(dense, y, MARKERS[variant], color=COLORS[variant], mec='white', ms=6, zorder=3,
                        label='Dense reference' if v_index == 0 and s_index == 0 else None)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=8.5, loc='left', color=INK)
        ax.set_xlabel('Value at development-selected threshold')
        ax.grid(axis='y', visible=False)
    ticks = [v * 4 + s for v in range(2) for s in range(3)]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([f'{VARIANTS[v]}, seed {s}' for v in VARIANTS for s in SEEDS], fontsize=7.5)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.6, 1.06))
    fig.tight_layout()
    save(fig, 'figure3_annotation_completeness')


def figure_stain(stain):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, dataset, title in zip(axes, ('dense_evaluation', 'apc_sparse_test'),
                                  ('(a) Dense SIPaKMeD evaluation', '(b) APCData held-out test (sparse labels)')):
        for v_index, variant in enumerate(VARIANTS):
            for s_index, seed in enumerate(SEEDS):
                rows = stain['models'][model_name(variant, seed)][dataset]
                for row in rows:
                    d = row['paired_difference']
                    delta = d['delta']['abnormal_recall']
                    lo, hi = d['paired_cluster_bootstrap_95ci']['abnormal_recall']
                    y = v_index * 12 + s_index * 4 + row['reference_index']
                    ax.plot([lo, hi], [y, y], color=COLORS[variant], lw=1.6, alpha=0.6)
                    ax.plot(delta, y, MARKERS[variant], color=COLORS[variant], ms=5, mec='white',
                            label=VARIANTS[variant] if s_index == 0 and row['reference_index'] == 0 else None)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=8.5, loc='left', color=INK)
        ax.set_xlabel('Change in abnormal recall\n(normalized − raw; paired 95% CI)')
        ax.grid(axis='y', visible=False)
    ticks = [v * 12 + s * 4 + 1 for v in range(2) for s in range(3)]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([f'{VARIANTS[v]}, seed {s}' for v in VARIANTS for s in SEEDS], fontsize=7.5)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'figure4_stain_reference_sensitivity')


def figure_calibration(models):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls='--')
        for seed, alpha in zip(SEEDS, (1.0, 0.7, 0.45)):
            cal = models[model_name(variant, seed)]['calibration']
            for kind, style in (('raw', ':'), ('scaled', '-')):
                bins = [b for b in cal[kind]['bins'] if b['count']]
                ax.plot([b['avg_confidence'] for b in bins], [b['accuracy'] for b in bins], style,
                        marker=MARKERS[variant], ms=3.5, lw=1.4, color=COLORS[variant], alpha=alpha)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean predicted confidence')
        ax.set_title(VARIANTS[variant] + ' (three seeds)', fontsize=8.5, loc='left', color=INK)
    axes[0].set_ylabel('Observed fraction correct')
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=MUTED, ls=':', lw=1.4, label='Raw confidence'),
            Line2D([], [], color=MUTED, ls='-', lw=1.4, label='Temperature-scaled'),
            Line2D([], [], color=MUTED, ls='--', lw=0.8, label='Perfect calibration')]
    fig.legend(handles=keys, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, 'figureS1_reliability')


def tables(models, paired, variability, manifest, completeness, stain, protocol):
    c, q = manifest['counts'], manifest['quarantine_counts']
    write_csv('table1_datasets', ['Source', 'Role', 'Train', 'Validation', 'Test', 'Excluded', 'Grouping unit'], [
        ['SIPaKMeD (Roboflow derivative)', 'Training/validation/test', c['sipakmed/train'], c['sipakmed/valid'], c['sipakmed/test'], q['sipakmed'], 'Field name + image-overlap component'],
        ['SIPaKMeD dense subset', 'Threshold development / evaluation', '', len(protocol['dense_development']), len(protocol['dense_evaluation']), '', 'Overlap component (withheld from training)'],
        ['APCData', 'Training/validation/test (combined variant)', c['apcdata/train'], c['apcdata/valid'], c['apcdata/test'], q['apcdata'], 'Specimen code + image-overlap component'],
        ['HMCHH-TCT-CellDet', 'External evaluation only', '', '', protocol['external']['fields'], '', 'Filename slide prefix'],
    ])
    rows = []
    for endpoint, label in (('dense_evaluation', 'Dense SIPaKMeD'), ('hmchh', 'HMCHH external')):
        for variant in VARIANTS:
            for seed in SEEDS:
                m = models[model_name(variant, seed)]
                r = m[endpoint]
                met, ci, n = r['metrics'], r['conditional_cluster_bootstrap_95ci'], r['counts']
                rows.append([label, VARIANTS[variant], seed, f'{m["threshold_from_development"]:.2f}',
                             f'{n["abnormal_tp"]}/{n["abnormal_reference_cells"]}',
                             fmt(met['abnormal_recall'], ci['abnormal_recall']),
                             fmt(met['abnormal_precision'], ci['abnormal_precision']),
                             fmt(met['abnormal_f1'], ci['abnormal_f1'], pct=False),
                             fmt(met['abnormal_false_detections_per_field'], ci['abnormal_false_detections_per_field'], pct=False),
                             fmt(met['localization_recall'], ci['localization_recall']),
                             fmt(met['localization_precision'], ci['localization_precision'])])
    sparse_rows = []
    for source, label in (('sipakmed', 'SIPaKMeD held-out test'), ('apcdata', 'APCData held-out test')):
        for variant in VARIANTS:
            for seed in SEEDS:
                r = models[model_name(variant, seed)]['sparse_test'][source]
                met, ci, n = r['metrics'], r['conditional_cluster_bootstrap_95ci'], r['counts']
                sparse_rows.append([label, VARIANTS[variant], seed, n['fields'], f'{n["abnormal_tp"]}/{n["abnormal_reference_cells"]}',
                                    fmt(met['abnormal_recall'], ci['abnormal_recall']),
                                    fmt(met['abnormal_precision'], ci['abnormal_precision']),
                                    fmt(met['localization_recall'], ci['localization_recall'])])
    write_csv('tableS2_sparse_heldout_tests', ['Evaluation', 'Training data', 'Seed', 'Fields', 'Abnormal TP/reference',
                                               'Abnormal recall % (95% CI)', 'Abnormal precision % (lower bound; 95% CI)',
                                               'Localization recall % (95% CI)'], sparse_rows)
    write_csv('table2_detection_performance',
              ['Evaluation', 'Training data', 'Seed', 'Threshold', 'Abnormal TP/reference', 'Abnormal recall % (95% CI)',
               'Abnormal precision % (95% CI)', 'Abnormal F1 (95% CI)', 'Unmatched abnormal predictions per field (95% CI)',
               'Localization recall % (95% CI)', 'Localization precision % (95% CI)'], rows)
    rows = []
    for seed in SEEDS:
        for endpoint, label in (('dense', 'Dense SIPaKMeD'), ('hmchh', 'HMCHH external')):
            d = paired[str(seed)][endpoint]
            for metric in ('abnormal_recall', 'abnormal_precision', 'abnormal_f1', 'abnormal_false_detections_per_field'):
                pct = metric != 'abnormal_f1' and 'per_field' not in metric
                rows.append([seed, label, metric, fmt(d['delta'][metric], d['paired_cluster_bootstrap_95ci'][metric], pct=pct), d['groups']])
    write_csv('table3_paired_differences', ['Seed', 'Evaluation', 'Metric', 'Combined − SIPaKMeD-only (95% CI)', 'Clusters'], rows)
    rows = []
    for variant in VARIANTS:
        for endpoint in ('dense_evaluation', 'hmchh'):
            for metric in ('abnormal_recall', 'abnormal_precision', 'abnormal_f1'):
                s = variability[variant][endpoint][metric]
                rows.append([VARIANTS[variant], endpoint, metric, f'{s["mean"]:.3f}', f'{s["sample_sd"]:.3f}', f'{s["min"]:.3f}', f'{s["max"]:.3f}'])
    write_csv('tableS6_seed_variability', ['Training data', 'Evaluation', 'Metric', 'Mean', 'SD', 'Min', 'Max'], rows)
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            cal = models[model_name(variant, seed)]['calibration']
            rows.append([VARIANTS[variant], seed, f'{cal["temperature"]:.3f}', cal['raw']['n_predictions'],
                         f'{cal["raw"]["ece"]:.3f}', f'{cal["scaled"]["ece"]:.3f}', f'{cal["raw"]["brier"]:.3f}',
                         f'{cal["scaled"]["brier"]:.3f}', f'{cal["raw"]["nll"]:.3f}', f'{cal["scaled"]["nll"]:.3f}'])
    write_csv('tableS4_calibration', ['Training data', 'Seed', 'Temperature', 'Predictions', 'ECE raw', 'ECE scaled',
                                     'Brier raw', 'Brier scaled', 'NLL raw', 'NLL scaled'], rows)
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            r = completeness['models'][model_name(variant, seed)]
            for ref in ('sparse_reference', 'dense_reference'):
                met, n = r[ref]['metrics'], r[ref]['counts']
                rows.append([VARIANTS[variant], seed, ref.split('_')[0], n['reference_cells'], n['abnormal_reference_cells'],
                             f'{met["localization_precision"]*100:.1f}', f'{met["localization_recall"]*100:.1f}',
                             f'{met["abnormal_precision"]*100:.1f}', f'{met["abnormal_recall"]*100:.1f}'])
    write_csv('tableS3_sparse_vs_dense_reference', ['Training data', 'Seed', 'Reference', 'Reference cells', 'Abnormal reference cells',
                                                   'Localization precision %', 'Localization recall %', 'Abnormal precision %', 'Abnormal recall %'], rows)
    rows = []
    for name, datasets in stain['models'].items():
        for dataset, entries in datasets.items():
            for row in entries:
                d = row['paired_difference']
                rows.append([name, dataset, row['reference_index'],
                             fmt(d['delta']['abnormal_recall'], d['paired_cluster_bootstrap_95ci']['abnormal_recall']),
                             fmt(d['delta']['abnormal_precision'], d['paired_cluster_bootstrap_95ci']['abnormal_precision'])])
    write_csv('tableS7_stain_reference_sensitivity', ['Model', 'Dataset', 'Reference set', 'Δ abnormal recall pp (95% CI)', 'Δ abnormal precision pp (95% CI)'], rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    models = load(V3 / 'evaluation_summary.json')['models']
    paired = load(V3 / 'paired_seed_comparisons.json')
    stain = load(V3 / 'stain_reference_sensitivity.json')
    completeness = load(PUB / 'results/annotation_completeness_secondary.json')
    manifest = load(ROOT / 'data/research_v3/manifest.json')
    protocol = load(V3 / 'evaluation_protocol.json')
    global N_EVAL
    N_EVAL = len(protocol['dense_evaluation'])
    figure_study_design(manifest, protocol)
    figure_results(models)
    figure_annotation(completeness)
    figure_stain(stain)
    figure_calibration(models)
    tables(models, paired['paired_by_seed'], paired['seed_variability'], manifest, completeness, stain, protocol)
    print('Figures and tables written to', PUB)


if __name__ == '__main__':
    main()
