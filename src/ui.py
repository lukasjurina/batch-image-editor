from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QSlider, QScrollArea, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os
from src.engine import GlitchEngine
from src.processor import BatchProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glitch-O-Matic")
        self.original_image = None
        self.proxy_image = None
        self.engine = GlitchEngine()
        self.processor = BatchProcessor(self.engine)
        
        self.params = {
            'warp': {'amplitude': 0, 'frequency': 5.0},
            'rgb': {'r_offset': 0, 'b_offset': 0},
            'scanlines': {'density': 2, 'opacity': 0.0}
        }
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left: Controls
        controls = QScrollArea()
        controls.setFixedWidth(250)
        controls_content = QWidget()
        self.controls_layout = QVBoxLayout(controls_content)
        
        self.btn_import = QPushButton("Import Image")
        self.btn_import.clicked.connect(self.load_image)
        self.controls_layout.addWidget(self.btn_import)
        
        # Warp Controls
        self.warp_amp_slider = self.add_slider("Warp Amplitude", 0, 100, 0, self.on_param_change)
        self.warp_freq_slider = self.add_slider("Warp Frequency (x10)", 1, 200, 50, self.on_param_change)
        
        # RGB Split Controls
        self.r_offset_slider = self.add_slider("Red Offset", -50, 50, 0, self.on_param_change)
        self.b_offset_slider = self.add_slider("Blue Offset", -50, 50, 0, self.on_param_change)
        
        # Scanlines Controls
        self.scan_opacity_slider = self.add_slider("Scanline Opacity (%)", 0, 100, 0, self.on_param_change)
        
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
        self.update_preview()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
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

    def update_preview(self):
        if self.proxy_image is None: return
        
        processed = self.proxy_image.copy()
        processed = self.engine.apply_sinusoidal_warp(processed, **self.params['warp'])
        processed = self.engine.apply_rgb_split(processed, **self.params['rgb'])
        processed = self.engine.apply_scanlines(processed, **self.params['scanlines'])
        
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
