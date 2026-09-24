"""Rebuild the HMCHH external reference from the original XML, abnormal cells only.

The earlier conversion (prepare_hmchh.py) assumed a single class after sampling
200 files. The source XML also contains organism/flora categories and polygon
items. That conversion kept every rectangle under the label "Abnormal"
(including Trichomonas, shift in flora and Actinomyces) and dropped polygon
annotations (including abnormal cells).

Here every XML item is parsed. Items named 异常 ("abnormal", including the
combined 异常，菌群失调) are kept; rectangles and polygon bounding boxes are
both used. Single-point annotations have no extent and are excluded and counted. Organism and flora categories are excluded and counted. Images are
the existing validation split of the prepared set; nothing is resampled.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

from prepare_combined_2class import place

SOURCE_IMAGES = Path('D:/pap_model/HMCHH_YOLO_prepared/valid/images')
XML_DIR = Path('D:/pap_model/Annotations_extracted/anno_copy')
OUT = Path('D:/pap_model/HMCHH_abnormal_reference_v3/valid')
ROOT = Path(__file__).resolve().parent
ABNORMAL = '异常'


def geometry(item):
    box = item.find('bndbox')
    if box is not None:
        return [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')], 'rectangle'
    polygon = item.find('polygon')
    values = {c.tag: float(c.text) for c in polygon}
    xs = [v for k, v in values.items() if k.startswith('x')]
    ys = [v for k, v in values.items() if k.startswith('y')]
    if len(xs) != len(ys):
        raise ValueError('Malformed polygon')
    if len(xs) < 3:
        return None, 'point'
    return [min(xs), min(ys), max(xs), max(ys)], 'polygon'


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} exists; frozen reference')
    (OUT / 'images').mkdir(parents=True)
    (OUT / 'labels').mkdir(parents=True)
    kept, excluded, geometry_counts, point_only = Counter(), Counter(), Counter(), Counter()
    fields = {}
    for image in sorted(SOURCE_IMAGES.glob('*.png')):
        root = ET.parse(XML_DIR / f'{image.stem}.xml').getroot()
        width, height = Image.open(image).size
        lines = []
        for item in root.iter('item'):
            name = item.findtext('name', '').strip()
            if ABNORMAL not in name:
                excluded[name] += 1
                continue
            box, kind = geometry(item)
            if box is None:
                point_only[name] += 1   # single-point annotation: no extent to score
                continue
            x1, y1, x2, y2 = box
            x1, x2 = max(0., min(x1, x2)), min(float(width), max(x1, x2))
            y1, y2 = max(0., min(y1, y2)), min(float(height), max(y1, y2))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Degenerate box in {image.name}')
            kept[name] += 1
            geometry_counts[kind] += 1
            lines.append(f'0 {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')
        place(image, OUT / 'images' / image.name)
        label = OUT / 'labels' / f'{image.stem}.txt'
        label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        fields[image.name] = {'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                              'label_sha256': hashlib.sha256(label.read_bytes()).hexdigest(), 'abnormal_cells': len(lines)}
    summary = {
        'source': 'HMCHH-TCT-CellDet (Zhang et al., Sci Data 2025; figshare 10.6084/m9.figshare.27901206), existing validation split',
        'rule': __doc__,
        'fields': len(fields), 'fields_without_abnormal_cells': sum(v['abnormal_cells'] == 0 for v in fields.values()),
        'abnormal_cells': sum(kept.values()), 'kept_by_name': dict(kept), 'kept_by_geometry': dict(geometry_counts),
        'excluded_by_name': dict(excluded),
        'abnormal_point_annotations_excluded': dict(point_only),
        'per_field': fields,
    }
    (OUT.parent / 'reference_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    tracked = ROOT / 'results/research_v3'
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / 'hmchh_reference_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_field', 'rule')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
