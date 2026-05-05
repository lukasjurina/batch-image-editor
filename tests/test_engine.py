import numpy as np
import pytest
from src.engine import GlitchEngine

def test_sinusoidal_warp_shape():
    engine = GlitchEngine()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.apply_sinusoidal_warp(img, amplitude=10, frequency=0.1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_rgb_split_shape():
    engine = GlitchEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_rgb_split(img, r_offset=5, b_offset=-5)
    assert result.shape == img.shape

def test_scanlines_shape():
    engine = GlitchEngine()
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.apply_scanlines(img, density=2, opacity=0.5)
    assert result.shape == img.shape
