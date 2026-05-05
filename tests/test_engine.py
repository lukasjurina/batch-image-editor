import numpy as np
import pytest
from engine import SecurityEngine

def test_sinusoidal_warp_shape():
    engine = SecurityEngine()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.apply_sinusoidal_warp(img, amplitude=10, frequency=0.1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_rgb_split_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_rgb_split(img, r_offset=5, b_offset=-5)
    assert result.shape == img.shape

def test_scanlines_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_scanlines(img, density=2, opacity=0.5)
    assert result.shape == img.shape

def test_pixel_sorting_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_pixel_sorting(img, threshold=128)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_bitcrush_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_bitcrush(img, levels=8)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_databending_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_databending(img, blocks=5, shift_max=10)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_micro_jitter_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_micro_jitter(img, intensity=0.1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_luminance_mask_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_luminance_mask(img, intensity=0.1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_checkerboard_mask_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_checkerboard_mask(img, density=4, opacity=0.3)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_line_interference_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_line_interference(img, count=10, thickness=1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_elastic_distortion_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_elastic_distortion(img, strength=5.0)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_salt_and_pepper_color_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_salt_and_pepper_color(img, intensity=0.05)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_ghosting_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_ghosting(img, offset=10, opacity=0.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_halftone_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_halftone(img, scale=5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_glitch_smear_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_glitch_smear(img, density=0.1, strength=20)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_local_inversion_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_local_inversion(img, count=5, size=10)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_rotation_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_rotation(img, angle=15.0)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_zoom_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_zoom(img, scale=1.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_perspective_warp_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_perspective_warp(img, intensity=0.1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_vignette_shape():
    engine = SecurityEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_vignette(img, strength=0.5)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
