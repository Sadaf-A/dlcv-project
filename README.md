# ShelfCheck AI - Retail Shelf Product Checker & Planogram Compliance

> Deep Learning-powered retail shelf monitoring system with Explainable AI for planogram compliance checking and inventory management.

![ShelfCheck AI Demo](https://img.shields.io/badge/Status-Production%20Ready-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)
## Dataset Link 
https://drive.google.com/file/d/1p3iDn5jatNN_PKyxYIljCLeLI3wgUs_V/view?usp=sharing

## 🎯 Overview

ShelfCheck AI is a comprehensive retail shelf monitoring solution that uses state-of-the-art computer vision models to detect changes in product placement, verify planogram compliance, and provide detailed inventory analysis. The system combines three powerful deep learning models with explainable AI features for transparent and reliable results.

### Key Features

- **Multi-Model Architecture**: YOLOv8, Faster R-CNN, and ResNet integration
- **Explainable AI**: Grad-CAM, feature visualization, confidence analysis
- **Real-time Web Interface**: Drag-and-drop image upload with live results
- **Planogram Compliance**: Automated before/after comparison with scoring
- **Model Benchmarking**: Performance comparison across different architectures
- **Comprehensive Reporting**: Detailed analysis with actionable insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask Backend  │    │   AI Models     │
│   (HTML/JS)     │◄──►│   (Python)       │◄──►│   (PyTorch)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│                      │                        │
├─ Image Upload        ├─ API Endpoints         ├─ YOLOv8 (Detection)
├─ Model Selection     ├─ Image Processing      ├─ Faster R-CNN
├─ Results Display     ├─ Model Orchestration   ├─ ResNet (Classification)
└─ Compliance Reports  └─ XAI Generation        └─ Explainable AI
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, CPU supported)
- 4GB+ RAM
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/shelfcheck-ai.git
   cd shelfcheck-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**
   ```bash
   python backend_server.py
   ```

4. **Open the web interface**
   Navigate to `http://localhost:5001` in your browser

## 📊 Dataset Format

ShelfCheck AI uses YOLO format annotations:

```
dataset/
├── train/
│   ├── images/          # Training images (.jpg, .png)
│   └── labels/          # labels (.txt)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
└── classes.txt          # Class names (one per line)
```

**Label format**: `class_id center_x center_y width height` (normalized 0-1)

## 🧠 Model Training

### YOLOv8 Training
```bash
python yolo_script.py \
    --dataset ./dataset
```

### Faster R-CNN Training
```bash
python r-cnn.py \
    --dataset ./dataset 
```

### ResNet Training
```bash
python resnet.py \
    --dataset ./dataset 
```

## 🔍 Explainable AI Features

ShelfCheck AI includes comprehensive XAI capabilities:

### Grad-CAM Visualization
- Highlights regions the model focuses on for decisions
- Class-specific activation maps
- Overlay visualizations on original images

### Feature Analysis
- Deep feature map visualization
- Layer-wise activation analysis
- Feature importance scoring

### Confidence Analysis
- Prediction confidence distributions
- Per-class confidence metrics
- Reliability assessment scores

### Usage Example
```python
from enhanced_rcnn_trainer import EnhancedRCNNTrainer

trainer = EnhancedRCNNTrainer('./dataset')
trainer.load_model_for_inference('./weights/best.pt')
xai_results = trainer.explain_single_image('./test_image.jpg')
```

## 🌐 API Reference

### Model Status
```http
GET /model_status
```
Returns the status of all loaded models.

### Complete Analysis
```http
POST /analyze_complete
Content-Type: application/json

{
  "before_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "after_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "model_choice": "yolo"
}
```

### Model Benchmarking
```http
POST /benchmark_models
Content-Type: application/json

{
  "test_image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

## 📈 Performance Metrics

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| YOLOv8n | ~15ms | 85% mAP | 6MB | Real-time detection |
| Faster R-CNN | ~200ms | 90% mAP | 160MB | High accuracy |
| ResNet50 | ~5ms | 92% acc | 25MB | Classification |

*Benchmarked on NVIDIA RTX 3080, 640x640 images*

## 🎛️ Configuration

### Backend Configuration (`backend_server.py`)
```python
DATASET_PATH = './dataset'  # Path to your dataset
MODEL_PATHS = {
    'yolo': './runs/detect/yolo_explainable/weights/best.pt',
    'rcnn': './runs/detect/rcnn_*/weights/best.pt',
    'resnet': './runs/classify/resnet_*/weights/best.pt'
}
```

### Training Configuration
Models support extensive configuration through command-line arguments:
- `--epochs`: Training epochs
- `--batch`: Batch size
- `--lr`: Learning rate
- `--img-size`: Input image size
- `--device`: Computation device (auto-detected)

## 🔧 Troubleshooting

### Common Issues

**Backend Connection Failed**
```bash
# Check if port 5001 is available
lsof -ti:5001 | xargs kill -9  # Kill existing process
python backend_server.py        # Restart server
```

**CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
python enhanced_yolo_trainer.py --batch 4 --device cpu
```

**Model Loading Errors**
```bash
# Check model paths and dataset structure
ls -la runs/detect/*/weights/
ls -la dataset/train/
```

### Performance Optimization

1. **Memory Management**
   - Use smaller batch sizes on limited hardware
   - Enable gradient checkpointing for large models
   - Clear cache regularly during training

2. **Speed Optimization**
   - Use TensorRT for inference acceleration
   - Implement model quantization
   - Batch multiple images for processing

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/yourusername/shelfcheck-ai.git
cd shelfcheck-ai
pip install -r requirements-dev.txt
pre-commit install
```

## 🙏 Acknowledgments

- **Team Members**: Avirup (Web Interface & LLM Integration), Lakshay (Planogram Logic), Sadaf (Object Detection Model Implementation)
- **Amrita Vishwa Vidyapeetham** for institutional support



---

<div align="center">
  Made with ❤️ by the ShelfCheck AI Team at Amrita Vishwa Vidyapeetham
</div>
