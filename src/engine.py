import numpy as np
import cv2
from numba import njit

class SecurityEngine:
    @staticmethod
    @njit
    def _sort_pixels(image_data, threshold):
        rows, cols, ch = image_data.shape
        for r in range(rows):
            c = 0
            while c < cols:
                # Simple brightness: (R+G+B)/3
                brightness = (int(image_data[r, c, 0]) + int(image_data[r, c, 1]) + int(image_data[r, c, 2])) // 3
                if brightness > threshold:
                    start = c
                    while c < cols:
                        brightness = (int(image_data[r, c, 0]) + int(image_data[r, c, 1]) + int(image_data[r, c, 2])) // 3
                        if brightness <= threshold: break
                        c += 1
                    end = c
                    
                    if end - start > 1:
                        segment = image_data[r, start:end]
                        n = end - start
                        for i in range(n):
                            for j in range(0, n - i - 1):
                                b1 = (int(segment[j, 0]) + int(segment[j, 1]) + int(segment[j, 2])) // 3
                                b2 = (int(segment[j+1, 0]) + int(segment[j+1, 1]) + int(segment[j+1, 2])) // 3
                                if b1 > b2:
                                    for k in range(3):
                                        tmp = segment[j, k]
                                        segment[j, k] = segment[j+1, k]
                                        segment[j+1, k] = tmp
                else:
                    c += 1
        return image_data

    def apply_pixel_sorting(self, image: np.ndarray, threshold: int) -> np.ndarray:
        if threshold >= 255: return image
        res = image.copy()
        return self._sort_pixels(res, threshold)

    def apply_bitcrush(self, image: np.ndarray, levels: int) -> np.ndarray:
        if levels >= 255 or levels <= 0: return image
        factor = 255 // levels
        return (image // factor * factor).astype(np.uint8)

    def apply_databending(self, image: np.ndarray, blocks: int, shift_max: int) -> np.ndarray:
        if blocks <= 0 or shift_max <= 0: return image
        rows, cols, _ = image.shape
        res = image.copy()
        for _ in range(blocks):
            y = np.random.randint(0, rows)
            h = np.random.randint(1, max(2, rows // 10))
            shift = np.random.randint(-shift_max, shift_max + 1)
            y_end = min(y + h, rows)
            res[y:y_end, :] = np.roll(res[y:y_end, :], shift, axis=1)
        return res

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

    @staticmethod
    @njit
    def _micro_jitter(image_data, intensity):
        rows, cols, ch = image_data.shape
        res = image_data.copy()
        for r in range(rows):
            for c in range(cols):
                if np.random.random() < intensity:
                    dr = np.random.randint(-1, 2)
                    dc = np.random.randint(-1, 2)
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        for k in range(ch):
                            tmp = res[r, c, k]
                            res[r, c, k] = res[nr, nc, k]
                            res[nr, nc, k] = tmp
        return res

    def apply_micro_jitter(self, image: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0: return image
        return self._micro_jitter(image, intensity)

    def apply_luminance_mask(self, image: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0: return image
        rows, cols, ch = image.shape
        noise = (np.random.randn(rows, cols, ch) * intensity * 255).astype(np.float32)
        
        # Luminance vector for BGR: Y = 0.114*B + 0.587*G + 0.299*R
        lum_vec = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        
        # Project noise to be luminance-neutral
        dot_products = np.sum(noise * lum_vec, axis=2, keepdims=True)
        lum_vec_sq_norm = np.sum(lum_vec**2)
        noise_projected = noise - (dot_products / lum_vec_sq_norm) * lum_vec
        
        res = image.astype(np.float32) + noise_projected
        return np.clip(res, 0, 255).astype(np.uint8)
