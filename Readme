
# CRFF-Net Installation Guide

This repository contains the CRFF-Net implementation for 3D object detection using 4D millimeter-wave radar and camera data.

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd CRFF-Net
```

---

## 2. Create Python Environment

It is recommended to use Python 3.9+ and a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu119
pip install torch-geometric
pip install numpy
pip install matplotlib
pip install opencv-python
```

> Note: Adjust PyTorch CUDA version according to your GPU.

---

## 4. Add Dataset

- Create folders inside `data/`:
  - `radar/` → 4D radar point clouds
  - `camera/` → camera images
  - `lidar/` → optional LiDAR point clouds for PCSR supervision

Place your VoD dataset files inside these folders.

---

## 5. Training

```bash
python train.py
```

- Trains the PCSR module and CRFF-Net detector
- Prints training loss per epoch

---

## 6. Testing / Evaluation

```bash
python test.py
```

- Runs forward pass on sample data
- Placeholder for AP/mAP evaluation and visualization

---

## 7. Utilities

- `utils.py` contains helper functions for:
  - Data loading
  - Preprocessing
  - Visualization (point clouds, BEV maps, detections)

---

## 8. Notes

- Ensure your GPU has sufficient memory (24GB recommended for full VoD dataset)
- Modify `train.py` batch size and dataset path as needed
- For full reproducibility, use VoD dataset and LiDAR-aligned dense radar point clouds
