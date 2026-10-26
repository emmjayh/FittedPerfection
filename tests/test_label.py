"""Label stage core: specificity, the separate skin layer, fusion, smoothing, hosts and sides."""
import numpy as np

from fittedperfection import label
from fittedperfection.meshes import FaceGraph

TORSO, SLEEVE, SKIN = label.CLASSES.index('torso'), label.CLASSES.index('sleeve'), label.CLASSES.index('skin')


def mask(shape, box):
    left, top, right, bottom = box
    return {'raster': f'{box}', 'bounds': list(box), 'confidence': 1.0, 'shape': [bottom - top, right - left]}, \
        np.ones((bottom - top, right - left), dtype=np.uint8)


def records(*specs):
    out, rasters = [], {}
    for cls, box in specs:
        record, raster = mask(None, box)
        record['class'] = cls
        record['raster'] = f'{cls}{box}'
        rasters[record['raster']] = raster
        out.append(record)
    return out, rasters


def test_smaller_garment_mask_wins_and_skin_never_erases_garment():
    recs, rasters = records(('torso', (0, 0, 10, 10)), ('sleeve', (2, 2, 5, 5)), ('skin', (0, 0, 4, 4)))
    winner, _, skin = label.paint_view(recs, rasters, 10)
    assert winner[3, 3] == SLEEVE and winner[8, 8] == TORSO
    assert skin[1, 1] == 1.0 and winner[1, 1] == TORSO          # skin recorded, torso kept


def test_a_view_seeing_the_face_without_the_label_counts_against_it():
    ids = np.zeros((4, 4), dtype=np.int64)                        # face 0 fills both views
    recs, rasters = records(('torso', (0, 0, 4, 4)))
    evidence, seen, covered = label.fuse({'a': ids, 'b': ids}, {'a': recs}, rasters, 1)
    assert seen[0] == 32 and covered[0] == 16 and np.isclose(evidence[0, TORSO], 0.5)


def chain(n):
    pairs = np.array([[i, i + 1] for i in range(n - 1)])
    return FaceGraph(pairs=pairs, seam=np.zeros(n - 1, bool), angle_deg=np.zeros(n - 1), face_count=n)


def test_smoothing_fills_a_face_no_mask_touched_from_its_neighbours():
    evidence = np.zeros((3, len(label.CLASSES)))
    evidence[[0, 2], SLEEVE] = 1.0
    covered = np.array([100.0, 0.0, 100.0])
    scores = label.smooth(evidence, covered, chain(3), 1.0, 0.1, 0.3, 40, 20)
    assert np.argmax(scores[1]) == SLEEVE and scores[1, SLEEVE] > 0.5


def test_unidentified_trim_takes_its_host_but_never_skin():
    graph = chain(5)
    labels = np.array(['sleeve', 'none', 'skin', 'none', 'skin'], dtype=object)
    changed, kept = label.host_unlabelled(labels, graph, np.ones(5), largest_share=0.5)
    assert labels.tolist() == ['sleeve', 'sleeve', 'skin', 'attachment', 'skin'] and changed == 2 and kept == []


def test_a_piece_spanning_both_sides_is_split_at_the_middle():
    graph = chain(4)
    labels = np.array(['footwear'] * 4, dtype=object)
    centres = np.array([[0.2, 0, 0], [0.1, 0, 0], [-0.1, 0, 0], [-0.2, 0, 0]])
    label.sides(labels, centres, graph, np.ones(4))
    assert labels.tolist() == ['footwear.left', 'footwear.left', 'footwear.right', 'footwear.right']
