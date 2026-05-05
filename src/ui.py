from PyQt6.QtWidgets import QMainWindow, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glitch-O-Matic")
        self.setCentralWidget(QLabel("Hello Glitch"))
