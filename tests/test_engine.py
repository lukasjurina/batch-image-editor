import numpy as np
import pytest
from src.engine import GlitchEngine

def test_sinusoidal_warp_shape():
    engine = GlitchEngine()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = engine.apply_sinusoidal_warp(img, amplitude=10, frequency=0.1)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
