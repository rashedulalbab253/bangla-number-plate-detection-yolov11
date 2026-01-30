# Bangla Number Plate Detection with YOLOv11

A high-precision Automatic License Plate Recognition (ALPR) system engineered for real-time detection and recognition of Bengali script in complex urban environments.

---

## 👨‍💻 Author
**Rashedul Albab**

---

## 🚀 Overview
This project leverages the **YOLOv11** architecture to provide a robust solution for identifying Bangladeshi vehicle number plates. It features a premium, custom-built web dashboard for real-time inference and analytics.

### Key Features
- **High-Precision Detection**: Optimized for diverse lighting and traffic density scanning.
- **Dynamic Session History**: Automated tracking and gallery view of recent recognition results.
- **Adjustable Sensitivity**: Real-time confidence threshold tuning for noise reduction and high-accuracy filtering.
- **Premium UI/UX**: Modern Glassmorphism dashboard powered by a high-performance FastAPI backend.
- **Multiple Model Support**: Benchmarked and compatible with PyTorch, ONNX, and OpenVINO formats.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd number-plate-detection
   ```

2. **Set up Virtual Environment**:
   ```bash
   py -3.12 -m venv env
   .\env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### 1. Launch the Professional Web Dashboard (Recommended)
This is the main application featuring a modern UI.
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:8000`.

### 2. Basic Command Line Detection
For processing a single image via CLI:
```bash
python detect.py --image "path/to/your/image.jpg"
```

### 3. Model Performance Benchmarking
To compare inference speeds across different formats (OpenVINO, ONNX, etc.):
```bash
python infertime_comparison_preprocessed.py
```

---

## 📊 Technical Performance
- **Model**: YOLOv11 (Nano/Standard weights)
- **Confidence**: Observed accuracy ~92%+ on validated datasets.
- **Optimizations**: Fully supports **Intel OpenVINO** for accelerated CPU inference on Intel-powered laptops.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
