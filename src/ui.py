from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QSlider, QScrollArea, QGroupBox, QListWidget,
                             QProgressDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os
from engine import SecurityEngine
from processor import BatchProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Secure Photo Editor")
        self.original_image = None
        self.proxy_image = None
        self.image_paths = []
        self.security_engine = SecurityEngine()
        self.processor = BatchProcessor(self.security_engine)
        
        self.params = {
            'warp': {'amplitude': 0, 'frequency': 5.0},
            'rgb': {'r_offset': 0, 'b_offset': 0},
            'scanlines': {'density': 2, 'opacity': 0.0},
            'pixel_sorting': {'threshold': 255},
            'bitcrush': {'levels': 255},
            'databending': {'blocks': 0, 'shift_max': 0},
            'micro_jitter': {'intensity': 0.0},
            'luminance_mask': {'intensity': 0.0},
            'checkerboard': {'density': 4, 'opacity': 0.0},
            'lines': {'count': 0, 'thickness': 1},
            'elastic': {'strength': 0.0},
            'color_noise': {'intensity': 0.0},
            'ghosting': {'offset': 0, 'opacity': 0.0},
            'halftone': {'scale': 1},
            'glitch_smear': {'density': 0.0, 'strength': 0},
            'local_inversion': {'count': 0, 'size': 0},
            'rotation': {'angle': 0.0},
            'zoom': {'scale': 1.0},
            'perspective': {'intensity': 0.0},
            'vignette': {'strength': 0.0}
        }
        
        self.init_ui()

    def add_slider_to_layout(self, layout, label, min_val, max_val, init_val, callback):
        l = QLabel(label)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setValue(init_val)
        s.valueChanged.connect(callback)
        layout.addWidget(l)
        layout.addWidget(s)
        return s

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left: Image List
        self.image_list = QListWidget()
        self.image_list.setFixedWidth(200)
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list)

        # Middle: Controls
        controls = QScrollArea()
        controls.setFixedWidth(250)
        controls_content = QWidget()
        self.controls_layout = QVBoxLayout(controls_content)
        
        self.btn_import = QPushButton("Import Images")
        self.btn_import.clicked.connect(self.load_images)
        self.controls_layout.addWidget(self.btn_import)
        
        self.btn_reset = QPushButton("Reset Parameters")
        self.btn_reset.clicked.connect(self.reset_parameters)
        self.controls_layout.addWidget(self.btn_reset)
        
        # Warp Group
        warp_group = QGroupBox("Warping Effects")
        warp_layout = QVBoxLayout()
        self.warp_amp_slider = self.add_slider_to_layout(warp_layout, "Amplitude", 0, 100, 0, self.on_param_change)
        self.warp_freq_slider = self.add_slider_to_layout(warp_layout, "Frequency (x10)", 1, 200, 50, self.on_param_change)
        warp_group.setLayout(warp_layout)
        self.controls_layout.addWidget(warp_group)

        # Geometry Group
        geom_group = QGroupBox("Geometry & Transforms")
        geom_layout = QVBoxLayout()
        self.rotation_slider = self.add_slider_to_layout(geom_layout, "Rotation Angle", -45, 45, 0, self.on_param_change)
        self.zoom_slider = self.add_slider_to_layout(geom_layout, "Zoom Scale (x100)", 100, 200, 100, self.on_param_change)
        self.persp_slider = self.add_slider_to_layout(geom_layout, "Perspective Intensity", 0, 100, 0, self.on_param_change)
        self.vignette_slider = self.add_slider_to_layout(geom_layout, "Vignette Strength", 0, 100, 0, self.on_param_change)
        geom_group.setLayout(geom_layout)
        self.controls_layout.addWidget(geom_group)
        
        # Glitch Group
        glitch_group = QGroupBox("Glitch & Color")
        glitch_layout = QVBoxLayout()
        self.r_offset_slider = self.add_slider_to_layout(glitch_layout, "Red Offset", -50, 50, 0, self.on_param_change)
        self.b_offset_slider = self.add_slider_to_layout(glitch_layout, "Blue Offset", -50, 50, 0, self.on_param_change)
        self.scan_opacity_slider = self.add_slider_to_layout(glitch_layout, "Scanline Opacity (%)", 0, 100, 0, self.on_param_change)
        self.sort_thresh_slider = self.add_slider_to_layout(glitch_layout, "Pixel Sort Threshold", 0, 255, 255, self.on_param_change)
        self.bitcrush_slider = self.add_slider_to_layout(glitch_layout, "Bitcrush Levels", 1, 255, 255, self.on_param_change)
        glitch_group.setLayout(glitch_layout)
        self.controls_layout.addWidget(glitch_group)

        # Data Group
        data_group = QGroupBox("Data Corruption")
        data_layout = QVBoxLayout()
        self.data_blocks_slider = self.add_slider_to_layout(data_layout, "Data Blocks", 0, 50, 0, self.on_param_change)
        self.data_shift_slider = self.add_slider_to_layout(data_layout, "Data Shift Max", 0, 100, 0, self.on_param_change)
        data_group.setLayout(data_layout)
        self.controls_layout.addWidget(data_group)
        
        # Anti-OCR Group
        ocr_group = QGroupBox("Anti-OCR Protections")
        ocr_layout = QVBoxLayout()
        self.jitter_slider = self.add_slider_to_layout(ocr_layout, "Micro-Jitter (%)", 0, 100, 0, self.on_param_change)
        self.lum_mask_slider = self.add_slider_to_layout(ocr_layout, "Luminance Mask (%)", 0, 100, 0, self.on_param_change)
        self.checker_opacity_slider = self.add_slider_to_layout(ocr_layout, "Checkerboard (%)", 0, 100, 0, self.on_param_change)
        self.lines_count_slider = self.add_slider_to_layout(ocr_layout, "Lines Count", 0, 100, 0, self.on_param_change)
        self.elastic_strength_slider = self.add_slider_to_layout(ocr_layout, "Elastic Strength", 0, 100, 0, self.on_param_change)
        self.color_noise_slider = self.add_slider_to_layout(ocr_layout, "Color Noise (%)", 0, 100, 0, self.on_param_change)
        ocr_group.setLayout(ocr_layout)
        self.controls_layout.addWidget(ocr_group)

        # Extreme Effects Group
        extreme_group = QGroupBox("Extreme Effects (OCR Confusion)")
        extreme_layout = QVBoxLayout()
        self.ghost_offset_slider = self.add_slider_to_layout(extreme_layout, "Ghost Offset", -100, 100, 0, self.on_param_change)
        self.ghost_opacity_slider = self.add_slider_to_layout(extreme_layout, "Ghost Opacity (%)", 0, 100, 0, self.on_param_change)
        self.halftone_scale_slider = self.add_slider_to_layout(extreme_layout, "Halftone Scale", 1, 50, 1, self.on_param_change)
        self.smear_density_slider = self.add_slider_to_layout(extreme_layout, "Smear Density (%)", 0, 100, 0, self.on_param_change)
        self.smear_strength_slider = self.add_slider_to_layout(extreme_layout, "Smear Strength", 0, 200, 0, self.on_param_change)
        self.inv_count_slider = self.add_slider_to_layout(extreme_layout, "Inversion Count", 0, 50, 0, self.on_param_change)
        self.inv_size_slider = self.add_slider_to_layout(extreme_layout, "Inversion Size", 0, 100, 0, self.on_param_change)
        extreme_group.setLayout(extreme_layout)
        self.controls_layout.addWidget(extreme_group)
        
        self.btn_export = QPushButton("Export All (Batch)")
        self.btn_export.clicked.connect(self.export_batch)
        self.controls_layout.addWidget(self.btn_export)
        
        controls.setWidget(controls_content)
        controls.setWidgetResizable(True)
        layout.addWidget(controls)

        # Right: Preview
        self.preview_label = QLabel("No image loaded")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label, 1)

    def on_param_change(self):
        self.params['warp']['amplitude'] = self.warp_amp_slider.value()
        self.params['warp']['frequency'] = self.warp_freq_slider.value() / 10.0
        self.params['rgb']['r_offset'] = self.r_offset_slider.value()
        self.params['rgb']['b_offset'] = self.b_offset_slider.value()
        self.params['scanlines']['opacity'] = self.scan_opacity_slider.value() / 100.0
        self.params['pixel_sorting']['threshold'] = self.sort_thresh_slider.value()
        self.params['bitcrush']['levels'] = self.bitcrush_slider.value()
        self.params['databending']['blocks'] = self.data_blocks_slider.value()
        self.params['databending']['shift_max'] = self.data_shift_slider.value()
        self.params['micro_jitter']['intensity'] = self.jitter_slider.value() / 500.0 # Max 20% jitter
        self.params['luminance_mask']['intensity'] = self.lum_mask_slider.value() / 500.0 # Max 0.2 intensity
        self.params['checkerboard']['opacity'] = self.checker_opacity_slider.value() / 500.0 # Max 0.2 opacity
        self.params['lines']['count'] = self.lines_count_slider.value()
        self.params['elastic']['strength'] = self.elastic_strength_slider.value() / 2.0 # Max 50.0 strength
        self.params['color_noise']['intensity'] = self.color_noise_slider.value() / 1000.0 # Max 10% noise
        self.params['ghosting']['offset'] = self.ghost_offset_slider.value()
        self.params['ghosting']['opacity'] = self.ghost_opacity_slider.value() / 100.0
        self.params['halftone']['scale'] = self.halftone_scale_slider.value()
        self.params['glitch_smear']['density'] = self.smear_density_slider.value() / 100.0
        self.params['glitch_smear']['strength'] = self.smear_strength_slider.value()
        self.params['local_inversion']['count'] = self.inv_count_slider.value()
        self.params['local_inversion']['size'] = self.inv_size_slider.value()
        self.params['rotation']['angle'] = self.rotation_slider.value()
        self.params['zoom']['scale'] = self.zoom_slider.value() / 100.0
        self.params['perspective']['intensity'] = self.persp_slider.value() / 500.0 # Max 0.2 intensity
        self.params['vignette']['strength'] = self.vignette_slider.value() / 50.0 # Max 2.0 strength
        self.update_preview()

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Images (*.png *.jpg *.jpeg)")
        if paths:
            self.image_paths = paths
            self.image_list.clear()
            for p in paths:
                self.image_list.addItem(os.path.basename(p))
            
            # Auto-load the first image
            self.image_list.setCurrentRow(0)
            self.load_selected_image(paths[0])

    def on_image_selected(self, item):
        row = self.image_list.row(item)
        if 0 <= row < len(self.image_paths):
            self.load_selected_image(self.image_paths[row])

    def load_selected_image(self, path):
        self.original_image = cv2.imread(path)
        if self.original_image is None: return
        # Create proxy (max 1000px)
        h, w = self.original_image.shape[:2]
        scale = min(1000 / w, 1000 / h)
        if scale < 1:
            self.proxy_image = cv2.resize(self.original_image, (int(w*scale), int(h*scale)))
        else:
            self.proxy_image = self.original_image.copy()
        self.update_preview()

    def reset_parameters(self):
        self.params = {
            'warp': {'amplitude': 0, 'frequency': 5.0},
            'rgb': {'r_offset': 0, 'b_offset': 0},
            'scanlines': {'density': 2, 'opacity': 0.0},
            'pixel_sorting': {'threshold': 255},
            'bitcrush': {'levels': 255},
            'databending': {'blocks': 0, 'shift_max': 0},
            'micro_jitter': {'intensity': 0.0},
            'luminance_mask': {'intensity': 0.0},
            'checkerboard': {'density': 4, 'opacity': 0.0},
            'lines': {'count': 0, 'thickness': 1},
            'elastic': {'strength': 0.0},
            'color_noise': {'intensity': 0.0},
            'ghosting': {'offset': 0, 'opacity': 0.0},
            'halftone': {'scale': 1},
            'glitch_smear': {'density': 0.0, 'strength': 0},
            'local_inversion': {'count': 0, 'size': 0},
            'rotation': {'angle': 0.0},
            'zoom': {'scale': 1.0},
            'perspective': {'intensity': 0.0},
            'vignette': {'strength': 0.0}
        }
        
        # Block signals to avoid multiple updates while resetting sliders
        self.warp_amp_slider.blockSignals(True)
        self.warp_freq_slider.blockSignals(True)
        self.rotation_slider.blockSignals(True)
        self.zoom_slider.blockSignals(True)
        self.persp_slider.blockSignals(True)
        self.vignette_slider.blockSignals(True)
        self.r_offset_slider.blockSignals(True)
        self.b_offset_slider.blockSignals(True)
        self.scan_opacity_slider.blockSignals(True)
        self.sort_thresh_slider.blockSignals(True)
        self.bitcrush_slider.blockSignals(True)
        self.data_blocks_slider.blockSignals(True)
        self.data_shift_slider.blockSignals(True)
        self.jitter_slider.blockSignals(True)
        self.lum_mask_slider.blockSignals(True)
        self.checker_opacity_slider.blockSignals(True)
        self.lines_count_slider.blockSignals(True)
        self.elastic_strength_slider.blockSignals(True)
        self.color_noise_slider.blockSignals(True)
        self.ghost_offset_slider.blockSignals(True)
        self.ghost_opacity_slider.blockSignals(True)
        self.halftone_scale_slider.blockSignals(True)
        self.smear_density_slider.blockSignals(True)
        self.smear_strength_slider.blockSignals(True)
        self.inv_count_slider.blockSignals(True)
        self.inv_size_slider.blockSignals(True)
        
        self.warp_amp_slider.setValue(0)
        self.warp_freq_slider.setValue(50)
        self.rotation_slider.setValue(0)
        self.zoom_slider.setValue(100)
        self.persp_slider.setValue(0)
        self.vignette_slider.setValue(0)
        self.r_offset_slider.setValue(0)
        self.b_offset_slider.setValue(0)
        self.scan_opacity_slider.setValue(0)
        self.sort_thresh_slider.setValue(255)
        self.bitcrush_slider.setValue(255)
        self.data_blocks_slider.setValue(0)
        self.data_shift_slider.setValue(0)
        self.jitter_slider.setValue(0)
        self.lum_mask_slider.setValue(0)
        self.checker_opacity_slider.setValue(0)
        self.lines_count_slider.setValue(0)
        self.elastic_strength_slider.setValue(0)
        self.color_noise_slider.setValue(0)
        self.ghost_offset_slider.setValue(0)
        self.ghost_opacity_slider.setValue(0)
        self.halftone_scale_slider.setValue(1)
        self.smear_density_slider.setValue(0)
        self.smear_strength_slider.setValue(0)
        self.inv_count_slider.setValue(0)
        self.inv_size_slider.setValue(0)
        
        self.warp_amp_slider.blockSignals(False)
        self.warp_freq_slider.blockSignals(False)
        self.rotation_slider.blockSignals(False)
        self.zoom_slider.blockSignals(False)
        self.persp_slider.blockSignals(False)
        self.vignette_slider.blockSignals(False)
        self.r_offset_slider.blockSignals(False)
        self.b_offset_slider.blockSignals(False)
        self.scan_opacity_slider.blockSignals(False)
        self.sort_thresh_slider.blockSignals(False)
        self.bitcrush_slider.blockSignals(False)
        self.data_blocks_slider.blockSignals(False)
        self.data_shift_slider.blockSignals(False)
        self.jitter_slider.blockSignals(False)
        self.lum_mask_slider.blockSignals(False)
        self.checker_opacity_slider.blockSignals(False)
        self.lines_count_slider.blockSignals(False)
        self.elastic_strength_slider.blockSignals(False)
        self.color_noise_slider.blockSignals(False)
        self.ghost_offset_slider.blockSignals(False)
        self.ghost_opacity_slider.blockSignals(False)
        self.halftone_scale_slider.blockSignals(False)
        self.smear_density_slider.blockSignals(False)
        self.smear_strength_slider.blockSignals(False)
        self.inv_count_slider.blockSignals(False)
        self.inv_size_slider.blockSignals(False)
        
        self.update_preview()

    def update_preview(self):
        if self.proxy_image is None: return
        
        processed = self.proxy_image.copy()
        processed = self.security_engine.apply_sinusoidal_warp(processed, **self.params['warp'])
        processed = self.security_engine.apply_rotation(processed, **self.params['rotation'])
        processed = self.security_engine.apply_zoom(processed, **self.params['zoom'])
        processed = self.security_engine.apply_perspective_warp(processed, **self.params['perspective'])
        processed = self.security_engine.apply_vignette(processed, **self.params['vignette'])
        processed = self.security_engine.apply_rgb_split(processed, **self.params['rgb'])
        processed = self.security_engine.apply_scanlines(processed, **self.params['scanlines'])
        processed = self.security_engine.apply_pixel_sorting(processed, **self.params['pixel_sorting'])
        processed = self.security_engine.apply_bitcrush(processed, **self.params['bitcrush'])
        processed = self.security_engine.apply_databending(processed, **self.params['databending'])
        processed = self.security_engine.apply_micro_jitter(processed, **self.params['micro_jitter'])
        processed = self.security_engine.apply_luminance_mask(processed, **self.params['luminance_mask'])
        processed = self.security_engine.apply_checkerboard_mask(processed, **self.params['checkerboard'])
        processed = self.security_engine.apply_line_interference(processed, **self.params['lines'])
        processed = self.security_engine.apply_elastic_distortion(processed, **self.params['elastic'])
        processed = self.security_engine.apply_salt_and_pepper_color(processed, **self.params['color_noise'])
        processed = self.security_engine.apply_ghosting(processed, **self.params['ghosting'])
        processed = self.security_engine.apply_halftone(processed, **self.params['halftone'])
        processed = self.security_engine.apply_glitch_smear(processed, **self.params['glitch_smear'])
        processed = self.security_engine.apply_local_inversion(processed, **self.params['local_inversion'])
        
        self.display_image(processed)

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w*ch, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def export_batch(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please import images first.")
            return
            
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir: return
        
        progress = QProgressDialog("Processing images...", "Cancel", 0, len(self.image_paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        for i, path in enumerate(self.image_paths):
            if progress.wasCanceled():
                break
            
            progress.setLabelText(f"Processing: {os.path.basename(path)}")
            progress.setValue(i)
            
            # Use a slightly modified processor call or just call it directly for better feedback
            # To keep it simple, we just call it once per file here
            try:
                self.processor.process_batch([path], output_dir, self.params)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process {path}:\n{str(e)}")
                break
        
        progress.setValue(len(self.image_paths))
        QMessageBox.information(self, "Success", f"Batch processing complete.\nFiles saved to: {output_dir}")
