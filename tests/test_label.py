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


SKIN_CLASS = label.CLASSES.index('skin')


def test_skin_recovery_takes_back_bare_patches_but_never_legwear_or_lone_pieces():
    # faces in a chain: skin | collar patch in the skin tone (off-colour for the collar) | dark collar x20 |
    # stocking in the skin tone | lone skin-tone collar face not joined to skin
    labels = np.array(['skin', 'collar'] + ['collar'] * 20 + ['pant_leg.left', 'collar'], dtype=object)
    colours = np.array([[200, 180, 180], [198, 182, 181]] + [[30, 30, 120]] * 20 + [[200, 181, 180], [200, 180, 182]],
                       dtype=float)
    scores = np.zeros((len(labels), len(label.CLASSES)))
    graph = chain(len(labels))
    pairs = graph.pairs[graph.pairs[:, 1] != len(labels) - 1]               # the last face touches nothing
    graph = FaceGraph(pairs=pairs, seam=np.zeros(len(pairs), bool), angle_deg=np.zeros(len(pairs)),
                      face_count=len(labels))
    changed = label.recover_skin(labels, scores, colours, graph, [200, 180, 180], 25.0, 0.35, 2.0)
    assert changed == 1 and labels[1] == 'skin' and labels[22] == 'pant_leg.left' and labels[23] == 'collar'


def test_skin_recovery_follows_sam_evidence_for_a_sliver_in_a_pale_garment():
    labels = np.array(['skin', 'sleeve.left', 'sleeve.left'] + ['torso'] * 20, dtype=object)
    colours = np.full((len(labels), 3), 218.0)                                # pale top on a pale mannequin
    scores = np.zeros((len(labels), len(label.CLASSES)))
    scores[1:3, SKIN_CLASS] = 0.5                                             # SAM saw skin on the sliver only
    changed = label.recover_skin(labels, scores, colours, chain(len(labels)), [218, 218, 218], 25.0, 0.35, 2.0)
    assert changed == 2 and (labels[3:] == 'torso').all()


def test_connected_to_follows_shared_vertices_within_the_member_faces():
    triangles = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5], [6, 7, 8]])
    welded = np.arange(9)
    member = np.array([True, True, True, True])
    seeds = np.array([True, False, False, False])
    assert label.connected_to(triangles, welded, member, seeds).tolist() == [True, True, True, False]
    member[1] = False                                                         # the bridge is not a member
    assert label.connected_to(triangles, welded, member, seeds).tolist() == [True, False, False, False]


def strip_graph(n):
    graph = chain(n)
    return graph


def test_a_sharp_dark_mark_on_a_light_part_is_a_print():
    labels = np.array(['torso'] * 80, dtype=object)
    colours = np.full((80, 3), 220.0)
    colours[30:40] = 20.0                                                   # a dark logo, sharp edges
    prints, pieces = label.detect_prints(labels, colours, chain(80), 60.0, 6, 120.0)
    assert pieces == 1 and prints[30:40].all() and prints.sum() == 10


def test_gradual_shading_and_specks_are_not_prints():
    labels = np.array(['torso'] * 80, dtype=object)
    shading = np.linspace(120, 255, 80)[:, None].repeat(3, axis=1)        # baked-in light: far from the median, no edge
    assert label.detect_prints(labels, shading, chain(80), 60.0, 6, 120.0)[0].sum() == 0
    speck = np.full((80, 3), 220.0)
    speck[10:12] = 20.0                                                     # two faces: too small
    assert label.detect_prints(labels, speck, chain(80), 60.0, 6, 120.0)[0].sum() == 0


def test_skin_and_footwear_are_never_prints():
    labels = np.array(['skin'] * 40 + ['footwear.left'] * 40, dtype=object)
    colours = np.full((80, 3), 220.0)
    colours[10:20] = colours[50:60] = 20.0
    assert label.detect_prints(labels, colours, chain(80), 60.0, 6, 120.0)[0].sum() == 0


def test_a_garment_part_in_the_skin_tone_is_skin_coloured_and_not_recovered():
    labels = np.array(['skin'] + ['torso'] * 60 + ['collar'] * 60, dtype=object)
    colours = np.array([[218, 218, 218]] * 61 + [[30, 30, 120]] * 60, dtype=float)   # white top, dark collar
    assert label.skin_coloured_parts(labels, colours, [218, 218, 218], 25.0) == {'torso'}
    scores = np.zeros((len(labels), len(label.CLASSES)))
    scores[1:61, SKIN_CLASS] = 0.9                                            # SAM calls the white top skin
    assert label.recover_skin(labels, scores, colours, chain(len(labels)), [218, 218, 218], 25.0, 0.35, 2.0) == 0


def test_skin_mirrored_onto_a_skin_coloured_part_takes_that_part_with_sides_swapped():
    # a white top on a white mannequin: the left back is skin, its mirror on the right is the top
    centres = np.array([[0.1, 0.0, 1.4], [-0.1, 0.0, 1.4], [0.1, 0.0, 0.9], [-0.1, 0.0, 0.9]] +
                       [[0.0, 0.1, 1.0 + 0.005 * i] for i in range(60)])
    labels = np.array(['skin', 'torso', 'skin', 'skin'] + ['torso'] * 60, dtype=object)
    colours = np.full((len(labels), 3), 218.0)
    changed = label.mirror_skin(labels, centres, colours, [218, 218, 218], 25.0)
    assert changed == 1 and labels[0] == 'torso' and labels[2] == 'skin' and labels[3] == 'skin'


def test_mirroring_swaps_the_side_and_skips_outfits_without_a_skin_coloured_part():
    centres = np.array([[0.1, 0.0, 1.4], [-0.1, 0.0, 1.4]] + [[0.0, 0.1, 1.0 + 0.005 * i] for i in range(60)])
    labels = np.array(['skin', 'sleeve.right'] + ['sleeve.right'] * 60, dtype=object)
    colours = np.full((len(labels), 3), 218.0)
    assert label.mirror_skin(labels.copy(), centres, colours * 0.2, [218, 218, 218], 25.0) == 0
    assert label.mirror_skin(labels, centres, colours, [218, 218, 218], 25.0) == 1 and labels[0] == 'sleeve.left'
