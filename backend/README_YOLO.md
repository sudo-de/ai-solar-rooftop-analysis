# YOLO Model Setup (YOLOv11/YOLOv12)

## Model File Location

The YOLO model files can be placed in any of these locations (checked in order):

1. `backend/models/yolo11n.pt` or `yolo12n.pt` (recommended)
2. `backend/yolo11n.pt` or `yolo12n.pt`
3. Project root: `yolo11n.pt` or `yolo12n.pt`
4. `models/yolo11n.pt` or `yolo12n.pt` (in project root)

## Automatic Download

If no model file is found, YOLO will automatically download `yolo11n.pt` (YOLOv11) from Ultralytics on first use.

## Model Versions

### YOLOv11 (Default - Recommended)
- `yolo11n.pt` (Nano): ~6.5 MB - **Default, most stable**
- Latest stable YOLO version with excellent compatibility
- Better detection for solar panels and photovoltaics
- Fully supported in ultralytics 8.3.232+

### YOLOv12 (Optional - Latest)
- `yolo12n.pt` (Nano): ~6.5 MB
- Latest YOLO version (requires ultralytics 8.3.232+)
- Enhanced accuracy and speed
- May have compatibility issues with older ultralytics versions

## Model File Sizes

- `yolo11n.pt` / `yolo12n.pt`: ~6.5 MB (Nano - fastest)
- `yolo11s.pt` / `yolo12s.pt`: ~22 MB (Small)
- `yolo11m.pt` / `yolo12m.pt`: ~52 MB (Medium)
- `yolo11l.pt` / `yolo12l.pt`: ~104 MB (Large)
- `yolo11x.pt` / `yolo12x.pt`: ~214 MB (Extra Large - most accurate)

## Using Custom Models

You can use other YOLO models by:

1. Downloading from [Ultralytics](https://github.com/ultralytics/ultralytics)
2. Placing in `backend/models/` directory
3. The service will automatically detect and use it

## Features

YOLOv11/YOLOv12 provides:
- **Enhanced Solar Panel Detection**: Better recognition of photovoltaic panels
- **Improved Hotspot Detection**: Advanced thermal anomaly detection in solar panels
- **Higher Accuracy**: Latest model architecture with improved performance
- **Faster Inference**: Optimized for speed while maintaining accuracy

## Requirements

- **ultralytics >= 8.3.232** (for YOLOv12 support)
- For YOLOv11: ultralytics >= 8.3.0

## Current Setup

The service will:
1. Try to find YOLOv11 models first (most stable)
2. Fall back to YOLOv12 if available
3. Auto-download YOLOv11 if nothing found
4. Gracefully handle compatibility issues

For best results with photovoltaics, use `yolo11n.pt` or `yolo12n.pt` (or larger models for higher accuracy).

