from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QSlider, QScrollArea, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glitch-O-Matic")
        self.original_image = None
        self.proxy_image = None
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
        
        # Slider example for Sinusoidal Warp
        self.add_slider("Warp Amplitude", 0, 100, 0, self.update_preview)
        
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

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.original_image = cv2.imread(path)
            # Create proxy (max 1000px)
            h, w = self.original_image.shape[:2]
            scale = 1000 / max(h, w)
            self.proxy_image = cv2.resize(self.original_image, (int(w*scale), int(h*scale)))
            self.update_preview()

    def update_preview(self):
        if self.proxy_image is None: return
        # Mock: just show proxy for now
        self.display_image(self.proxy_image)

    def display_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w*ch, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
