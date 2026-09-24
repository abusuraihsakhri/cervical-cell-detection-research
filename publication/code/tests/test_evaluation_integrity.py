import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from evaluation.taxonomy import binary_index_map, native_index_map
from evaluation.confusion_matrix import per_class_precision_recall, match_predictions_to_gt
from evaluation.calibration import fit_temperature_scale, apply_temperature
from annotation.evaluate_dense import parse_labelstudio_export
from prepare_research_v2 import group_id
from evaluation.research_v2 import field_counts, ratios
from evaluation.research_v2_synthesis import paired_difference


class EvaluationIntegrityTests(unittest.TestCase):
    def test_actual_checkpoint_order_and_reordering(self):
        names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        self.assertEqual(binary_index_map(names), {0: 1, 1: 1, 2: 0, 3: 0, 4: 0})
        self.assertEqual(binary_index_map(list(reversed(names))), {0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
        self.assertEqual(native_index_map(['Abnormal', 'Normal'], ['Normal', 'Abnormal']), {0: 1, 1: 0})

    def test_unknown_taxonomy_rejected(self):
        with self.assertRaises(ValueError):
            binary_index_map(['unknown'])
        with self.assertRaises(ValueError):
            native_index_map(['Dyskeratotic'], ['NILM'])

    def test_support_includes_missed_ground_truth(self):
        metrics = per_class_precision_recall(np.array([[3, 1, 6], [2, 4, 4], [1, 0, 0]]), ['Normal', 'Abnormal'])
        self.assertEqual(metrics['Normal']['support'], 10)
        self.assertAlmostEqual(metrics['Normal']['recall'], .3)

    def test_duplicate_detection_not_double_true_positive(self):
        box = [0, 0, 1, 1]
        pairs, fp, fn = match_predictions_to_gt([box, box], [0, 0], [box], [0])
        self.assertEqual((len(pairs), len(fp), len(fn)), (1, 1, 0))

    def test_temperature_minimizes_binary_nll(self):
        records = [(0.9, i < 6) for i in range(10)]
        temperature = fit_temperature_scale(records)
        self.assertAlmostEqual(apply_temperature(records, temperature)[0][0], .6, places=4)

    def test_unreviewed_export_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root / 'tasks.json'
            export.write_text(json.dumps([{'data': {'image_name': 'x.jpg'}, 'predictions': [{'result': []}]}]))
            with self.assertRaises(ValueError):
                parse_labelstudio_export(export, root, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_specimen_group_ignores_crop_number_and_spacing(self):
        self.assertEqual(group_id('apcdata', '0532 ASCH K 26177 18 (4).jpg'), group_id('apcdata', '0543 ASCH K26177 18 (15).jpg'))
        self.assertIsNone(group_id('sipakmed', 'cell_2_0_0_500_500_jpg.rf.abc.jpg'))

    def test_abnormal_counts_include_missed_and_false_cells(self):
        box = [0, 0, .4, .4]
        data = {'a': {'group': 'g', 'boxes': [{'cls': 1, 'bbox': box}, {'cls': 1, 'bbox': [.6,.6,1,1]}]}}
        predictions = {'a': [{'cls': 1, 'conf': .8, 'bbox': box}, {'cls': 1, 'conf': .7, 'bbox': box}]}
        counts = field_counts(data, predictions, .5)['a']['counts']
        self.assertEqual(counts, [1, 2, 2, 1, 2, 2, 1])
        self.assertEqual(ratios(counts)['abnormal_recall'], .5)
        self.assertEqual(ratios(counts)['abnormal_false_detections_per_field'], 1.)

    def test_frozen_group_and_hash_disjointness(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'data/research_v2/manifest.json'
        if not path.exists():
            self.skipTest('Frozen local dataset unavailable')
        manifest = json.loads(path.read_text())
        for key in ('group', 'sha256'):
            membership = {}
            for record in manifest['records']:
                membership.setdefault(record[key], set()).add(record['split'])
            self.assertTrue(all(len(splits) == 1 for splits in membership.values()))
        protocol = json.loads((root / 'results/research_v2/evaluation_protocol.json').read_text())
        self.assertFalse(set(protocol['dense_development']) & set(protocol['dense_evaluation']))
        dense_groups = {group_id('sipakmed', name) for name in protocol['dense_development'] + protocol['dense_evaluation']} - {None}
        self.assertFalse(dense_groups & {r['group'] for r in manifest['records'] if r['split'] in ('train', 'valid', 'test')})

    def test_paired_bootstrap_preserves_identical_models(self):
        fields = {'a': {'group': 'one', 'counts': [2, 3, 4, 1, 2, 2, 1]},
                  'b': {'group': 'two', 'counts': [1, 2, 3, 1, 1, 2, 1]}}
        result = paired_difference(fields, fields, iterations=50)
        self.assertTrue(all(v == 0 for v in result['delta'].values()))
        self.assertTrue(all(v == [0., 0.] for v in result['paired_cluster_bootstrap_95ci'].values()))

    def test_unknown_annotation_class_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'x.jpg').write_bytes(b'image-not-opened-before-validation')
            task = {'data': {'image_name': 'x.jpg'}, 'annotations': [{'completed_by': 1, 'created_at': '2026-09-19', 'result': [
                {'type': 'rectanglelabels', 'value': {'rectanglelabels': ['maybe'], 'x': 1, 'y': 1, 'width': 2, 'height': 2}}
            ]}]}
            (root/'tasks.json').write_text(json.dumps([task]))
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                parse_labelstudio_export(root/'tasks.json', root, root/'out')
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
