import numpy as np
import pytest

from openscope.data import Tile, tile_grid


def test_keeps_only_tiles_with_enough_tissue():
    mask = np.zeros((128, 128), dtype=bool)
    mask[:64, :64] = True
    tiles = tile_grid(mask, tile_size=32)
    assert {(t.x, t.y) for t in tiles} == {(0, 0), (32, 0), (0, 32), (32, 32)}
    assert all(t.tissue_fraction == 1.0 for t in tiles)


def test_scale_maps_to_full_resolution():
    mask = np.zeros((8, 8), dtype=bool)
    mask[4:, 4:] = True
    assert tile_grid(mask, tile_size=4, scale=32) == [Tile(128, 128, 128, 1.0)]


def test_partial_tiles_follow_the_threshold():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :1] = True  # 25% tissue
    assert tile_grid(mask, tile_size=4, min_tissue=0.5) == []
    assert len(tile_grid(mask, tile_size=4, min_tissue=0.25)) == 1


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        tile_grid(np.zeros((4, 4, 3), dtype=bool), tile_size=2)
    with pytest.raises(ValueError):
        tile_grid(np.zeros((4, 4), dtype=bool), tile_size=0)
