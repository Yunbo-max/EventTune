import numpy as np

from eventttt.qwen import class_balanced_weights
from eventttt.schemas import Sample


def sample(index, label, label_id):
    return Sample(
        sample_id=str(index),
        event_id="event",
        tile_id=str(index),
        pre_image="pre.png",
        post_image="post.png",
        label=label,
        label_id=label_id,
        dataset="test",
    )


def test_class_balanced_weights_equalize_total_class_mass():
    rows = [sample(0, "intact", 0)]
    rows += [sample(index, "damaged", 1) for index in range(1, 4)]
    rows += [sample(index, "destroyed", 2) for index in range(4, 9)]

    weights = class_balanced_weights(rows).numpy()
    totals = [
        weights[[row.label_id == label_id for row in rows]].sum()
        for label_id in range(3)
    ]
    assert np.allclose(totals, [1.0, 1.0, 1.0])

