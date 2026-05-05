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
                    
                    n = end - start
                    if n > 1:
                        # Shell sort is much faster than bubble sort and easy to implement in Numba
                        gap = n // 2
                        while gap > 0:
                            for i in range(gap, n):
                                temp = image_data[r, start + i].copy()
                                temp_b = (int(temp[0]) + int(temp[1]) + int(temp[2])) // 3
                                j = i
                                while j >= gap:
                                    prev_b = (int(image_data[r, start + j - gap, 0]) + 
                                             int(image_data[r, start + j - gap, 1]) + 
                                             int(image_data[r, start + j - gap, 2])) // 3
                                    if prev_b > temp_b:
                                        for k in range(3):
                                            image_data[r, start + j, k] = image_data[r, start + j - gap, k]
                                        j -= gap
                                    else:
                                        break
                                for k in range(3):
                                    image_data[r, start + j, k] = temp[k]
                            gap //= 2
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

    @staticmethod
    @njit
    def _checkerboard_mask(image_data, density, opacity):
        rows, cols, ch = image_data.shape
        res = image_data.copy().astype(np.float32)
        for r in range(rows):
            for c in range(cols):
                if ((r // density) + (c // density)) % 2 == 0:
                    for k in range(ch):
                        res[r, c, k] *= (1.0 - opacity)
        return res.astype(np.uint8)

    def apply_checkerboard_mask(self, image: np.ndarray, density: int, opacity: float) -> np.ndarray:
        if opacity <= 0 or density <= 0: return image
        return self._checkerboard_mask(image, density, opacity)

    def apply_line_interference(self, image: np.ndarray, count: int, thickness: int) -> np.ndarray:
        if count <= 0: return image
        rows, cols, _ = image.shape
        res = image.copy()
        for _ in range(count):
            pt1 = (np.random.randint(0, cols), np.random.randint(0, rows))
            pt2 = (np.random.randint(0, cols), np.random.randint(0, rows))
            
            # Pick a random color from the image to blend in
            sample_pt = (np.random.randint(0, cols), np.random.randint(0, rows))
            color = tuple([int(c) for c in image[sample_pt[1], sample_pt[0]]])
            
            cv2.line(res, pt1, pt2, color, thickness)
        return res

    def apply_elastic_distortion(self, image: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0: return image
        rows, cols, _ = image.shape
        
        # Create displacement field
        dx = (np.random.rand(rows, cols).astype(np.float32) * 2 - 1) * strength
        dy = (np.random.rand(rows, cols).astype(np.float32) * 2 - 1) * strength
        
        # Smooth the displacement field for "organic" look
        kernel_size = int(max(rows, cols) * 0.1)
        if kernel_size % 2 == 0: kernel_size += 1
        dx = cv2.GaussianBlur(dx, (kernel_size, kernel_size), 0)
        dy = cv2.GaussianBlur(dy, (kernel_size, kernel_size), 0)
        
        map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
        map_x = (map_x.astype(np.float32) + dx)
        map_y = (map_y.astype(np.float32) + dy)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    @staticmethod
    @njit
    def _salt_and_pepper_color(image_data, intensity):
        rows, cols, ch = image_data.shape
        res = image_data.copy()
        for r in range(rows):
            for c in range(cols):
                if np.random.random() < intensity:
                    for k in range(ch):
                        res[r, c, k] = np.random.randint(0, 256)
        return res

    def apply_salt_and_pepper_color(self, image: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0: return image
        return self._salt_and_pepper_color(image, intensity)

    def apply_ghosting(self, image: np.ndarray, offset: int, opacity: float) -> np.ndarray:
        if opacity <= 0 or offset == 0: return image
        rows, cols, _ = image.shape
        res = image.copy()
        ghost = np.roll(image, offset, axis=1)
        return cv2.addWeighted(res, 1.0 - opacity, ghost, opacity, 0)

    def apply_halftone(self, image: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1: return image
        rows, cols, ch = image.shape
        res = np.full((rows, cols, ch), 255, dtype=np.uint8) # White background
        
        for r in range(0, rows, scale):
            for c in range(0, cols, scale):
                r_end = min(r + scale, rows)
                c_end = min(c + scale, cols)
                window = image[r:r_end, c:c_end]
                avg_color = np.mean(window, axis=(0, 1))
                
                # Darker = larger circle
                intensity = 1.0 - (np.mean(avg_color) / 255.0)
                radius = int((scale / 2) * intensity)
                
                if radius > 0:
                    center = (c + scale // 2, r + scale // 2)
                    color = tuple([int(x) for x in avg_color])
                    cv2.circle(res, center, radius, color, -1, cv2.LINE_AA)
        return res

    def apply_glitch_smear(self, image: np.ndarray, density: float, strength: int) -> np.ndarray:
        if density <= 0 or strength <= 0: return image
        rows, cols, _ = image.shape
        res = image.copy()
        for r in range(rows):
            if np.random.random() < density:
                x_start = np.random.randint(0, max(1, cols - strength))
                x_end = min(cols, x_start + strength)
                res[r, x_start:x_end] = res[r, x_start]
        return res

    def apply_local_inversion(self, image: np.ndarray, count: int, size: int) -> np.ndarray:
        if count <= 0 or size <= 0: return image
        rows, cols, _ = image.shape
        res = image.copy()
        for _ in range(count):
            cx, cy = np.random.randint(0, cols), np.random.randint(0, rows)
            y_start, y_end = max(0, cy - size), min(rows, cy + size)
            x_start, x_end = max(0, cx - size), min(cols, cx + size)
            
            roi = res[y_start:y_end, x_start:x_end]
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx - x_start, cy - y_start), size, 255, -1)
            roi[mask == 255] = 255 - roi[mask == 255]
        return res

    def apply_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0: return image
        rows, cols, _ = image.shape
        center = (cols // 2, rows // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

    def apply_zoom(self, image: np.ndarray, scale: float) -> np.ndarray:
        if scale <= 1.0: return image
        rows, cols, _ = image.shape
        
        # Calculate crop area
        new_h, new_w = int(rows / scale), int(cols / scale)
        start_r, start_c = (rows - new_h) // 2, (cols - new_w) // 2
        
        cropped = image[start_r:start_r+new_h, start_c:start_c+new_w]
        return cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_LINEAR)

    def apply_perspective_warp(self, image: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0: return image
        rows, cols, _ = image.shape
        
        # Define source and destination points
        offset = intensity * min(rows, cols) * 0.2
        src_pts = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_pts = np.float32([
            [offset, offset], 
            [cols-1-offset, 0], 
            [0, rows-1], 
            [cols-1-offset, rows-1-offset]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

    def apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0: return image
        rows, cols, _ = image.shape
        
        # Generate Gaussian kernels for vignette
        kernel_x = cv2.getGaussianKernel(cols, cols / (strength * 2))
        kernel_y = cv2.getGaussianKernel(rows, rows / (strength * 2))
        kernel = kernel_y * kernel_x.T
        
        # Normalize and apply as mask
        mask = kernel / kernel.max()
        res = image.astype(np.float32)
        for i in range(3):
            res[:, :, i] *= mask
            
        return res.astype(np.uint8)
