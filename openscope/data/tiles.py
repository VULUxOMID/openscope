"""Tiling: choose the square tiles of a slide or frame that contain enough tissue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Tile:
    """A square tile. ``x`` and ``y`` are the top-left corner in output pixels."""

    x: int
    y: int
    size: int
    tissue_fraction: float


def tile_grid(
    mask: np.ndarray,
    tile_size: int,
    stride: Optional[int] = None,
    min_tissue: float = 0.5,
    scale: int = 1,
) -> List[Tile]:
    """Return the tiles whose tissue fraction is at least ``min_tissue``.

    ``mask`` is a boolean tissue mask, often computed on a low-resolution level
    of a whole-slide image. ``tile_size`` and ``stride`` are in mask pixels.
    ``scale`` converts mask pixels to output pixels: a mask made at 1/32 of the
    slide's resolution uses ``scale=32`` to return full-resolution tiles.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    stride = stride or tile_size
    if stride <= 0:
        raise ValueError("stride must be positive")
    height, width = mask.shape
    tiles: List[Tile] = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            fraction = float(mask[y : y + tile_size, x : x + tile_size].mean())
            if fraction >= min_tissue:
                tiles.append(Tile(x * scale, y * scale, tile_size * scale, fraction))
    return tiles
