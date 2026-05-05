from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QSlider, QScrollArea, QGroupBox, QListWidget)
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
            'luminance_mask': {'intensity': 0.0}
        }
        
        self.init_ui()

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
        
        # Warp Controls
        self.warp_amp_slider = self.add_slider("Warp Amplitude", 0, 100, 0, self.on_param_change)
        self.warp_freq_slider = self.add_slider("Warp Frequency (x10)", 1, 200, 50, self.on_param_change)
        
        # RGB Split Controls
        self.r_offset_slider = self.add_slider("Red Offset", -50, 50, 0, self.on_param_change)
        self.b_offset_slider = self.add_slider("Blue Offset", -50, 50, 0, self.on_param_change)
        
        # Scanlines Controls
        self.scan_opacity_slider = self.add_slider("Scanline Opacity (%)", 0, 100, 0, self.on_param_change)
        
        # Pixel Sorting
        self.sort_thresh_slider = self.add_slider("Pixel Sort Threshold", 0, 255, 255, self.on_param_change)
        
        # Bitcrush
        self.bitcrush_slider = self.add_slider("Bitcrush Levels", 1, 255, 255, self.on_param_change)
        
        # Databending
        self.data_blocks_slider = self.add_slider("Data Blocks", 0, 50, 0, self.on_param_change)
        self.data_shift_slider = self.add_slider("Data Shift Max", 0, 100, 0, self.on_param_change)
        
        # Anti-OCR
        self.jitter_slider = self.add_slider("Micro-Jitter Intensity (%)", 0, 100, 0, self.on_param_change)
        self.lum_mask_slider = self.add_slider("Luminance Mask (%)", 0, 100, 0, self.on_param_change)
        
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

    def add_slider(self, label, min_val, max_val, init_val, callback):
        group = QGroupBox(label)
        l = QVBoxLayout()
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setValue(init_val)
        s.valueChanged.connect(callback)
        l.addWidget(s)
        group.setLayout(l)
        self.controls_layout.addWidget(group)
        return s

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
            'luminance_mask': {'intensity': 0.0}
        }
        
        # Block signals to avoid multiple updates while resetting sliders
        self.warp_amp_slider.blockSignals(True)
        self.warp_freq_slider.blockSignals(True)
        self.r_offset_slider.blockSignals(True)
        self.b_offset_slider.blockSignals(True)
        self.scan_opacity_slider.blockSignals(True)
        self.sort_thresh_slider.blockSignals(True)
        self.bitcrush_slider.blockSignals(True)
        self.data_blocks_slider.blockSignals(True)
        self.data_shift_slider.blockSignals(True)
        self.jitter_slider.blockSignals(True)
        self.lum_mask_slider.blockSignals(True)
        
        self.warp_amp_slider.setValue(0)
        self.warp_freq_slider.setValue(50)
        self.r_offset_slider.setValue(0)
        self.b_offset_slider.setValue(0)
        self.scan_opacity_slider.setValue(0)
        self.sort_thresh_slider.setValue(255)
        self.bitcrush_slider.setValue(255)
        self.data_blocks_slider.setValue(0)
        self.data_shift_slider.setValue(0)
        self.jitter_slider.setValue(0)
        self.lum_mask_slider.setValue(0)
        
        self.warp_amp_slider.blockSignals(False)
        self.warp_freq_slider.blockSignals(False)
        self.r_offset_slider.blockSignals(False)
        self.b_offset_slider.blockSignals(False)
        self.scan_opacity_slider.blockSignals(False)
        self.sort_thresh_slider.blockSignals(False)
        self.bitcrush_slider.blockSignals(False)
        self.data_blocks_slider.blockSignals(False)
        self.data_shift_slider.blockSignals(False)
        self.jitter_slider.blockSignals(False)
        self.lum_mask_slider.blockSignals(False)
        
        self.update_preview()

    def update_preview(self):
        if self.proxy_image is None: return
        
        processed = self.proxy_image.copy()
        processed = self.security_engine.apply_sinusoidal_warp(processed, **self.params['warp'])
        processed = self.security_engine.apply_rgb_split(processed, **self.params['rgb'])
        processed = self.security_engine.apply_scanlines(processed, **self.params['scanlines'])
        processed = self.security_engine.apply_pixel_sorting(processed, **self.params['pixel_sorting'])
        processed = self.security_engine.apply_bitcrush(processed, **self.params['bitcrush'])
        processed = self.security_engine.apply_databending(processed, **self.params['databending'])
        processed = self.security_engine.apply_micro_jitter(processed, **self.params['micro_jitter'])
        processed = self.security_engine.apply_luminance_mask(processed, **self.params['luminance_mask'])
        
        self.display_image(processed)

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w*ch, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def export_batch(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images for Batch Processing", "", "Images (*.png *.jpg *.jpeg)")
        if not files: return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir: return
        
        self.processor.process_batch(files, output_dir, self.params)
        print(f"Batch processing complete. Files saved to {output_dir}")
