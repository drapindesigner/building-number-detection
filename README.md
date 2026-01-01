# Building Number Detection and Recognition System

A comprehensive machine perception system for detecting, segmenting, and recognizing building numbers from real-world images using deep learning and computer vision techniques.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

##  Project Overview

This project implements a complete end-to-end pipeline for automated building number extraction from photographs. The system is designed to handle various signage types, including plaques, wall-mounted lettering, and atypical designs commonly found on university campuses.

### Key Features

- **Robust Detection**: YOLOv8-based detector with 99% precision and 85% recall
- **Intelligent Segmentation**: Hybrid segmentation combining learned and heuristic approaches
- **Character Recognition**: Custom CNN achieving 88.9% accuracy on EMNIST benchmark
- **End-to-End Pipeline**: Fully integrated system from image input to text output
- **Highly Configurable**: Centralized configuration system with command-line overrides
- **Production Ready**: Handles negative cases, edge cases, and achieves <1 minute runtime

## Architecture

The system consists of four main components:

1. **Task 1 - Building Number Detection**: Localizes building number regions in images using YOLOv8
2. **Task 2 - Character Segmentation**: Extracts individual characters using hybrid segmentation
3. **Task 3 - Character Recognition**: Classifies characters using a custom CNN
4. **Task 4 - Integrated Pipeline**: Combines all components for end-to-end processing

```
Input Image → Detection → Segmentation → Recognition → Output Text
              (YOLOv8)   (Hybrid CNN)   (Custom CNN)   (Validated)
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended)
- conda or virtualenv

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/building-number-detection.git
cd building-number-detection

# Create and activate conda environment
conda env create -f comp3007.yml
conda activate comp3007

# Or use pip with requirements.txt
pip install -r requirements.txt
```

### Basic Usage

The system provides a unified interface through `assignment.py`:

```bash
# Task 1: Detect building numbers
python assignment.py task1 /path/to/images

# Task 2: Segment characters from detected regions
python assignment.py task2 /path/to/building_number_crops

# Task 3: Recognize individual characters
python assignment.py task3 /path/to/character_crops

# Task 4: Complete end-to-end pipeline
python assignment.py task4 /path/to/images
```

### Example Workflow

```bash
# Run complete pipeline on validation images
python assignment.py task4 ./validation/

# Check results
ls output/task4/
# → img1.txt, img2.txt, img3.txt, img4.txt (img5.txt correctly omitted for negative)
```

## Performance Metrics

### Detection (Task 1)
- **Precision**: 0.99
- **Recall**: 0.85
- **mAP@50**: 0.85
- **mAP@50-95**: 0.67

### Segmentation (Task 2)
- **Precision**: 0.99
- **Recall**: 0.99
- **mAP@50**: 0.99
- **mAP@50-95**: 0.81

### Recognition (Task 3)
- **Overall Accuracy**: 88.9% (on EMNIST balanced)
- **Character Set**: 0-9, A-D (14 classes)
- **Test Set Size**: 6,800 samples

### Runtime
- **Execution Time**: <38 seconds for 5 images (on RTX 3080)
- **Validation**: 4/4 positive images correctly processed, 1/1 negative correctly rejected

## Technical Details

### Datasets Used

1. **Building Number Detection**:
   - Roboflow export: ~3.2k annotated images
   - Custom Curtin campus captures (masked)

2. **Character Segmentation**:
   - Detection crops + dedicated character detector dataset (~15k labeled glyphs)

3. **Character Recognition**:
   - EMNIST Balanced: 112k samples (digits + letters)
   - Custom campus glyphs: 1.9k curated samples
   - SVHN/USPS digits for domain randomization
   - Roboflow character classification corpus

### Model Training

**Detection (YOLOv8n)**:
- 100 epochs, image size 640×640, batch size 16
- Augmentations: mosaic, HSV shifts, random perspective
- Best checkpoint: epoch 88

**Segmentation (YOLOv8n)**:
- 150 epochs, image size 320×320, batch size 32
- Additional augmentations: motion blur, contrast jitter

**Recognition (Custom CNN)**:
- 20 epochs, batch size 256
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Custom oversampling for class balance

### Configuration

The system uses a centralized configuration file (`config.txt`) with support for command-line overrides:

```bash
# Override detector confidence threshold
python assignment.py task1 /path/to/images --det_conf 0.35

# Use hybrid segmentation mode
python assignment.py task2 /path/to/crops --seg_mode hybrid

# Adjust recognition confidence
python assignment.py task3 /path/to/chars --rec_confidence_threshold 0.8
```

## Project Structure

```
project-root/
├── assignment.py          # Main entry point
├── task1.py              # Detection implementation
├── task2.py              # Segmentation implementation
├── task3.py              # Recognition implementation
├── task4.py              # Integrated pipeline
├── config.txt            # Configuration file
├── requirements.txt      # Python dependencies
├── perception/           # Core library
│   ├── detection/       # YOLOv8 detector
│   ├── segmentation/    # Hybrid segmenter
│   ├── recognition/     # CNN recognizer
│   ├── pipeline.py      # End-to-end pipeline
│   ├── config.py        # Configuration management
│   └── utils.py         # Utility functions
├── data/                 # Pre-trained weights and datasets
├── docs/                 # Documentation and training artifacts
│   ├── report.md        # Detailed technical report
│   └── runs/            # Training results and visualizations
└── output/              # Task outputs (generated)
    ├── task1/           # Detected building number crops
    ├── task2/           # Segmented character images
    ├── task3/           # Character recognition results
    └── task4/           # Final text outputs
```

## Example Results

### Input Image → Detection → Segmentation → Recognition

```
Input: campus_building.jpg
↓
Detection: Building number region localized
↓
Segmentation: Individual characters extracted (4 characters)
↓
Recognition: "200B" (with confidence scores)
↓
Output: 200B.txt
```

## Advanced Configuration

### Key Configuration Parameters

**Detection Settings**:
- `det_conf`: Confidence threshold (default: 0.25)
- `det_iou`: IoU threshold for NMS (default: 0.45)
- `det_img`: Inference resolution (default: 640)
- `det_crop_expand`: Crop expansion factor (default: 1.1)

**Segmentation Settings**:
- `seg_mode`: Segmentation mode (heuristic/learned/hybrid)
- `seg_min_area`: Minimum segment area (default: 100)
- `seg_padding`: Character crop padding (default: 2)

**Recognition Settings**:
- `rec_confidence_threshold`: Minimum confidence (default: 0.75)
- `rec_temperature`: Softmax temperature (default: 1.0)
- `rec_charset`: Character set (default: "0123456789ABCD")

See `docs/report.md` for comprehensive configuration guide.

## Development

### Training New Models

```bash
# Train detection model
python train_task1.py --epochs 100 --batch 16 --imgsz 640

# Train segmentation model
python train_task2.py --epochs 150 --batch 32 --imgsz 320

# Train recognition model
python train_task3.py --epochs 20 --batch 256
```

### Running Tests

```bash
# Validate on assignment test set
python assignment.py task4 ./validation/

# Check output consistency
diff -r output/task4/ expected_output/
```

## Known Limitations

- **Character Set**: Currently limited to `0123456789ABCD`
- **Single Detection**: Assumes one building number per image
- **Specular Highlights**: Type III signage with highlights may produce low confidence
- **Recognition Accuracy**: 88.9% leaves room for improvement on stylized characters

## Future Improvements

1. **Language Model Integration**: Add rule-based validation for building number sequences
2. **Semi-Supervised Learning**: Reduce reliance on labeled data
3. **Model Distillation**: Create lighter models for deployment (e.g., MobileNetV3)
4. **Automated Testing**: Regression test suite for configuration changes
5. **Extended Character Set**: Support additional letters and special characters

## References

1. [Building Numbers Dataset](https://universe.roboflow.com/mp-data/building-numbers-f6mly-vj3up) - Roboflow Universe
2. [Number Character Detection Dataset](https://universe.roboflow.com/mp-assignment-8jzdp/number-character-detection-3rtu3) - Roboflow Universe
3. [Number Character Classification Dataset](https://universe.roboflow.com/mp-assignment-8jzdp/number-character-classification-gnp7s) - Roboflow Universe
4. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
5. [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) - Extended MNIST

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../LICENSE) file for details.

## Author

**Mohammad Saif Ul Haq**
- University: Curtin University
- Course: COMP3007 - Machine Perception

---

