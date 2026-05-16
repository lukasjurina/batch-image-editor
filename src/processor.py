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
            if 'rotation' in params:
                img = self.security_engine.apply_rotation(img, **params['rotation'])
            if 'zoom' in params:
                img = self.security_engine.apply_zoom(img, **params['zoom'])
            if 'perspective' in params:
                img = self.security_engine.apply_perspective_warp(img, **params['perspective'])
            if 'vignette' in params:
                img = self.security_engine.apply_vignette(img, **params['vignette'])
            img = self.security_engine.apply_rgb_split(img, **params['rgb'])
            if 'scanlines' in params:
                img = self.security_engine.apply_scanlines(img, **params['scanlines'])
            if 'pixel_sorting' in params:
                img = self.security_engine.apply_pixel_sorting(img, **params['pixel_sorting'])
            if 'bitcrush' in params:
                img = self.security_engine.apply_bitcrush(img, **params['bitcrush'])
            if 'databending' in params:
                img = self.security_engine.apply_databending(img, **params['databending'])
            if 'micro_jitter' in params:
                img = self.security_engine.apply_micro_jitter(img, **params['micro_jitter'])
            if 'luminance_mask' in params:
                img = self.security_engine.apply_luminance_mask(img, **params['luminance_mask'])
            if 'checkerboard' in params:
                img = self.security_engine.apply_checkerboard_mask(img, **params['checkerboard'])
            if 'lines' in params:
                img = self.security_engine.apply_line_interference(img, **params['lines'])
            if 'elastic' in params:
                img = self.security_engine.apply_elastic_distortion(img, **params['elastic'])
            if 'color_noise' in params:
                img = self.security_engine.apply_salt_and_pepper_color(img, **params['color_noise'])
            if 'ghosting' in params:
                img = self.security_engine.apply_ghosting(img, **params['ghosting'])
            if 'halftone' in params:
                img = self.security_engine.apply_halftone(img, **params['halftone'])
            if 'glitch_smear' in params:
                img = self.security_engine.apply_glitch_smear(img, **params['glitch_smear'])
            if 'local_inversion' in params:
                img = self.security_engine.apply_local_inversion(img, **params['local_inversion'])
            
            name = os.path.basename(path)
            cv2.imwrite(os.path.join(output_dir, f"{name}"), img)
