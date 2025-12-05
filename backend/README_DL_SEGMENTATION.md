# Deep Learning Roof Segmentation Models

This document describes the deep learning-based roof segmentation models implemented in the system.

## Available Models

### 1. U-Net
- **Architecture**: Encoder-decoder with skip connections
- **Encoder**: ResNet34 (ImageNet pretrained)
- **Advantages**: 
  - Good for precise boundary detection
  - Skip connections preserve fine details
  - Fast inference
- **Best for**: High-resolution roof boundary detection

### 2. DeepLabv3+
- **Architecture**: Atrous convolution with ASPP module
- **Encoder**: ResNet50 (ImageNet pretrained)
- **Advantages**:
  - Multi-scale feature extraction
  - Better handling of objects at different scales
  - Robust to varying roof sizes
- **Best for**: Complex roofs with multiple structures

### 3. HRNet (High-Resolution Network)
- **Architecture**: Multi-resolution parallel branches
- **Encoder**: HRNet-W18 (ImageNet pretrained)
- **Advantages**:
  - Maintains high-resolution representations throughout
  - Excellent spatial detail preservation
  - State-of-the-art accuracy
- **Best for**: Maximum accuracy requirements

## Installation

```bash
pip install torch torchvision segmentation-models-pytorch
```

## Usage

### Basic Usage

```python
from yolo_service.segmentation import segment_roof
from PIL import Image

# Load image
image = Image.open("roof_image.jpg")

# Use U-Net
result = segment_roof(image, method="unet")

# Use DeepLabv3+
result = segment_roof(image, method="deeplabv3plus")

# Use HRNet
result = segment_roof(image, method="hrnet")
```

### With Custom Trained Model

```python
# Load your trained model
result = segment_roof(
    image, 
    method="unet",
    dl_model_path="./models/custom_unet_roof.pth"
)
```

### Fallback to Computer Vision

If PyTorch is not available, the system automatically falls back to computer vision methods:

```python
# Automatically uses enhanced_canny if DL unavailable
result = segment_roof(image, method="unet")  # Falls back to enhanced_canny
```

## Model Training (Future)

To train custom models for roof segmentation:

1. **Prepare Dataset**:
   - Collect roof images with pixel-level annotations
   - Format: Images + corresponding mask images

2. **Training Script** (to be implemented):
   ```python
   # Example training pipeline
   from yolo_service.segmentation import UNetSegmentation
   # ... training code ...
   ```

3. **Save Model**:
   ```python
   torch.save(model.state_dict(), "models/unet_roof.pth")
   ```

## Performance Comparison

| Model | Accuracy | Speed | Memory | Best Use Case |
|-------|----------|-------|--------|--------------|
| Enhanced Canny | Medium | Fast | Low | Quick analysis |
| U-Net | High | Medium | Medium | General purpose |
| DeepLabv3+ | Very High | Medium | High | Complex roofs |
| HRNet | Highest | Slow | High | Maximum accuracy |

## Notes

- Models use ImageNet pretrained weights by default (general purpose)
- For best results, train on roof-specific dataset
- GPU recommended for faster inference
- Models automatically resize input to 512x512 for inference
- Output masks are resized back to original image dimensions

