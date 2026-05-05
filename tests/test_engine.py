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
