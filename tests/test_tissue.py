import numpy as np

from openscope.data import otsu_threshold, rgb_to_saturation, tissue_mask


def synthetic_field(seed: int = 0) -> np.ndarray:
    """Near-white glass with a pink-purple 'tissue' square in the middle, plus noise."""
    rng = np.random.default_rng(seed)
    img = np.full((128, 128, 3), 236, dtype=np.float64)
    img[32:96, 32:96] = (196, 96, 160)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_saturation_of_gray_is_zero_and_of_color_is_high():
    gray = np.full((2, 2, 3), 128, dtype=np.uint8)
    pink = np.full((2, 2, 3), (200, 100, 150), dtype=np.uint8)
    assert np.allclose(rgb_to_saturation(gray), 0.0)
    assert np.allclose(rgb_to_saturation(pink), 0.5)


def test_otsu_splits_two_clusters():
    values = np.concatenate([np.full(500, 0.1), np.full(500, 0.6)])
    threshold = otsu_threshold(values)
    assert 0.1 < threshold <= 0.6


def test_tissue_mask_finds_the_square():
    mask = tissue_mask(synthetic_field())
    inside = mask[40:88, 40:88].mean()
    outside = np.concatenate([mask[:24].ravel(), mask[104:].ravel()]).mean()
    assert inside > 0.99
    assert outside < 0.01


def test_empty_glass_has_no_tissue():
    rng = np.random.default_rng(1)
    glass = np.clip(np.full((64, 64, 3), 236.0) + rng.normal(0, 3, (64, 64, 3)), 0, 255).astype(np.uint8)
    assert tissue_mask(glass).mean() < 0.01
