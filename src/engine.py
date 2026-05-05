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
