# Secure Photo Editor

A desktop tool for batch photo processing with anti-OCR and geometric effects.

## Installation

1. **Setup environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install PyQt6 opencv-python numpy numba pytest
   ```

## Usage

1. **Run the app:**
   ```bash
   python src/main.py
   ```

2. **Steps:**
   - Click **"Import Images"** to load one or more photos.
   - Adjust effects using the sliders in the right panel (real-time preview).
   - Click **"Reset Parameters"** to clear all effects.
   - Click **"Export All (Batch)"** to process and save all imported images to a folder.

## Testing
```bash
export PYTHONPATH=src
pytest tests/test_engine.py
```
