import os
import cv2
from src.engine import GlitchEngine

class BatchProcessor:
    def __init__(self, engine: GlitchEngine):
        self.engine = engine

    def process_batch(self, input_paths, output_dir, params):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for path in input_paths:
            img = cv2.imread(path)
            if img is None:
                continue
                
            # Apply stack
            img = self.engine.apply_sinusoidal_warp(img, **params['warp'])
            img = self.engine.apply_rgb_split(img, **params['rgb'])
            if 'scanlines' in params:
                img = self.engine.apply_scanlines(img, **params['scanlines'])
            
            name = os.path.basename(path)
            cv2.imwrite(os.path.join(output_dir, f"glitch_{name}"), img)
