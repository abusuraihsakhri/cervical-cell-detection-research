import unittest
import xml.etree.ElementTree as ET

import numpy as np

from prepare_research_v3 import Components
from prepare_hmchh_abnormal_reference import geometry, ABNORMAL
from publication.scripts.detect_field_overlap import confirm
from evaluation.sweep_operating_thresholds import evaluate_at_threshold


def item(xml):
    return ET.fromstring(f'<item>{xml}</item>')


class ResearchV3Tests(unittest.TestCase):
    def test_overlap_chains_become_one_component(self):
        comps = Components()
        comps.union('a', 'b')
        comps.union('c', 'd')
        comps.union('b', 'c')
        self.assertEqual(len({comps.find(x) for x in 'abcd'}), 1)
        self.assertNotEqual(comps.find('a'), comps.find('e'))

    def test_hmchh_rectangle_polygon_and_point(self):
        box, kind = geometry(item('<name>异常</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>'))
        self.assertEqual((box, kind), ([10., 20., 30., 40.], 'rectangle'))
        box, kind = geometry(item('<name>异常</name><polygon><x1>5</x1><y1>9</y1><x2>15</x2><y2>1</y2><x3>8</x3><y3>20</y3></polygon>'))
        self.assertEqual((box, kind), ([5., 1., 15., 20.], 'polygon'))
        self.assertEqual(geometry(item('<name>异常</name><polygon><x1>5</x1><y1>9</y1></polygon>')), (None, 'point'))

    def test_hmchh_organisms_are_not_abnormal_cells(self):
        for name in ('滴虫', '菌群失调', '念珠菌', '放线菌'):
            self.assertNotIn(ABNORMAL, name)
        self.assertIn(ABNORMAL, '异常，菌群失调')

    def test_overlap_confirmation_rejects_empty_features(self):
        self.assertEqual(confirm(([], None), ([], None), None), (0, 0.0, 0.0))

    def test_threshold_tie_selects_lowest(self):
        gt = {'f': {'boxes': [{'cls': 1, 'bbox': [0, 0, .5, .5]}]}}
        preds = {'f': [{'bbox': [0, 0, .5, .5], 'conf': .9, 'cls': 1}]}
        sweep = [evaluate_at_threshold(gt, preds, t / 100) for t in range(1, 66)]
        self.assertEqual(max(sweep, key=lambda v: v['abnormal_f1'])['conf'], .01)


if __name__ == '__main__':
    unittest.main()
