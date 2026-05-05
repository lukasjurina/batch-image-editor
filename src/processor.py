import os
import cv2
from engine import SecurityEngine

class BatchProcessor:
    def __init__(self, engine: SecurityEngine):
        self.security_engine = engine

    def process_batch(self, input_paths, output_dir, params):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for path in input_paths:
            img = cv2.imread(path)
            if img is None:
                continue
                
            # Apply stack
            img = self.security_engine.apply_sinusoidal_warp(img, **params['warp'])
            img = self.security_engine.apply_rgb_split(img, **params['rgb'])
            if 'scanlines' in params:
                img = self.security_engine.apply_scanlines(img, **params['scanlines'])
            if 'pixel_sorting' in params:
                img = self.security_engine.apply_pixel_sorting(img, **params['pixel_sorting'])
            if 'bitcrush' in params:
                img = self.security_engine.apply_bitcrush(img, **params['bitcrush'])
            if 'databending' in params:
                img = self.security_engine.apply_databending(img, **params['databending'])
            
            name = os.path.basename(path)
            cv2.imwrite(os.path.join(output_dir, f"secure_{name}"), img)
