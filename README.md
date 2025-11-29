# Shrimp Seed Detection and Counting

## Overview
This project implements an automated system for detecting and counting shrimp seeds/larvae using computer vision techniques. Built as a Streamlit-based web application, it provides shrimp farmers and hatchery operators with an easy-to-use tool for real-time counting, eliminating the need for manual processes. The system is designed to improve population density monitoring, inventory management, and feeding optimization.

## Vision Behind the Project
The inspiration for this project stems from Bhimavaram, a village in Andhra Pradesh, India, where aquaculture is a primary source of livelihood. Farmers in this region often rely on manual counting of shrimp seeds before purchase, which is both time-consuming and prone to errors. This project was initiated with the vision of creating a scalable solution that could eventually evolve into a mobile application, making it even more accessible to farmers in rural areas. While this version is implemented as a web-based tool, it lays the groundwork for future enhancements that align with this vision.

## Features

- **Automated Detection**: Identifies individual shrimp seeds in images or video streams using advanced computer vision models.
    
- **Accurate Counting**: Provides precise counts with minimal margin of error (93-96% accuracy).
    
- **Batch Processing**: Processes multiple images simultaneously for large-scale analysis.
    
- **User-Friendly Web Interface**: Built using Streamlit, offering an intuitive platform for image upload and result visualization.
    
- **Report Generation**: Exports detailed reports with count statistics and visualizations.
    
- **Scalable Design**: Can be extended to support mobile applications or cloud-based solutions in the future.

## Technology Stack
- **Programming Language**: Python 3.11
- **Computer Vision**: OpenCV, TensorFlow/PyTorch, detectron2 (from facebook)
- **Image Processing**: NumPy, SciPy
- **Model Architecture**: YOLOv5/Faster R-CNN for object detection
- **UI Framework**: Streamlit/Flask (web interface)

## Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Webcam or digital camera for live counting

### Setup
```bash
# Clone the repository
git clone https://github.com/rishendra-manne/shrimp_seeds_detection.git
cd shrimp_seeds_detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Web Interface
```bash
# Start the web application
python app.py
```
Then open your browser and navigate to `http://localhost:8501`

Special installation note for detectron2:
Since detectron2 is not available directly via pip, it should be installed separately with:
```bash
# For CUDA-enabled GPU:
pip install 'gitthttps: //github.com/facebookresearch/detectron2.git'
# For CPU-only:
pip install 'git+https://github.com/facebookresearch/detectron2.git@main#subdirectory=projects/PointRend'
```

## Model Training

If you need to train the model on your specific prawn species or environment:

```bash
# Train the detection model with custom data
python src/pipelines/training_pipeline.py --data_path ./data/custom_dataset --epochs 50 --batch_size 16 --model yolov5_custom.pt

```

## Performance Metrics
- **Counting Accuracy**: 93-96% under optimal conditions
- **Processing Speed**: 2-3 seconds for a image
- **Minimum Detectable Size**: 0.5mm larvae
- **Optimal Water Clarity**: 80%+ transparency of thin layer

## Troubleshooting

### Common Issues
1. **Poor Detection Accuracy**
   - Ensure proper lighting conditions
   - Check water clarity and background contrast
   - Adjust detection threshold in config.yml

2. **Slow Processing**
   - Reduce input image resolution
   - Enable GPU acceleration
   - Use batch processing for large datasets

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

