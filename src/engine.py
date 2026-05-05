import numpy as np
import cv2

class GlitchEngine:
    def apply_sinusoidal_warp(self, image: np.ndarray, amplitude: int, frequency: float) -> np.ndarray:
        if amplitude == 0: return image
        rows, cols, ch = image.shape
        map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Matematika vlnění v ose X
        map_x = map_x + amplitude * np.sin(2 * np.pi * map_y * frequency / rows)
        
        # Remap pomocí OpenCV (linear interpolation pro hladkost)
        distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
        return distorted

    def apply_rgb_split(self, image: np.ndarray, r_offset: int, b_offset: int) -> np.ndarray:
        rows, cols, _ = image.shape
        res = image.copy()
        
        if r_offset != 0:
            res[:, :, 2] = np.roll(image[:, :, 2], r_offset, axis=1) # Red v OpenCV je index 2
        if b_offset != 0:
            res[:, :, 0] = np.roll(image[:, :, 0], b_offset, axis=1) # Blue v OpenCV je index 0
            
        return res

    def apply_scanlines(self, image: np.ndarray, density: int, opacity: float) -> np.ndarray:
        if opacity == 0: return image
        rows, cols, _ = image.shape
        res = image.copy().astype(float)
        for i in range(0, rows, max(1, density)):
            res[i, :, :] *= (1 - opacity)
        return res.astype(np.uint8)
