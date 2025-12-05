# Solar Rooftop Analysis System Architecture

**Full-Stack Application**: React + Vite Frontend + FastAPI Backend
```
                ┌─────────────────────────────┐
                │                             │
                │   INPUT IMAGERY             │
                │                             │
                │  (Upload)                   │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │  PREPROCESS IMAGE           │
                │                             │
                │  - Crop around building     │
                │  - Normalize resolution     │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ ROOF SEGMENTATION MODEL     │
                │                             │
                │ (U-Net / DeepLabv3+ / HRNet)│
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ ROOF MASK                   │
                │                             │
                │  (Edges + Roof area)        │
                │                             │
                └──────────────┬──────────────┘
                               ▼
     ┌────────────────────────────────────────────────┐
     │                                                │
     │  DETECT OBSTRUCTIONS (YOLOv9/YOLOv10/YOLv11    │
     │  Mask R-CNN)                                   │
     │                                                │
     │  - Chimneys                                    │
     │  - Skylights                                   │
     │  - Water tanks                                 │
     │  - HVAC units                                  │
     │  - Vents / Dishes                              │
     │                                                │
     └──────────────────────────┬─────────────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ OBSTRUCTION MASKS           │
                │                             │
                │ (per-object segmentation)   │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ SUBTRACT OBSTRUCTIONS       │
                │                             │
                │usable_area = roof - objects │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ USABLE ROOF AREA POLYGON    │
                │                             │
                │  - Convert mask → polygon   │
                │  - Smooth edges             │
                │  - Calculate m² area        │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ SOLAR LAYOUT OPTIMIZER      │
                │                             │
                │  - Panel placement          │
                │  - Orientation              │
                │  - Spacing rules            │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │ SOLAR ENERGY COMPUTATION    │
                │                             │
                │  - kWp capacity             │
                │  - kWh/year estimate        │
                │  - Shading factor           │
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │                             │
                │   FINAL OUTPUT              │
                │                             │
                │  - Usable roof area         │
                │  - Panel layout map         │
                │  - Generation estimate      │
                │  - Savings & payback        │
                │                             │
                └─────────────────────────────┘
```

## Implementation Details

### 1. PREPROCESS IMAGE
**Status**: ✅ **Implemented & Enhanced**

**Location**: `backend/yolo_service/preprocessing.py`, `backend/main.py`, `frontend/src/components/AnalysisForm.tsx`

**Frontend Preprocessing**:
- File type validation: PNG, JPG, JPEG only
- File size validation: Maximum 10MB per file
- Drag & drop support for image uploads
- Multiple file selection support
- Image preview before upload

**Backend Preprocessing** (Enhanced):

1. **Crop around building**:
   - Uses YOLO model to detect buildings in the image
   - **Enhanced detection**: Searches for multiple building-related keywords (building, house, roof, structure, construction, residential, commercial, etc.)
   - **Smart merging**: If multiple buildings are detected close together, merges their bounding boxes
   - Finds the largest building detection or merged building area
   - Automatically crops to building area with smart padding:
     - Uses 20% of building size as padding
     - Minimum 50px padding to ensure context
     - Ensures crop box stays within image bounds
   - Falls back to full image if no building detected
   - **Function**: `detect_building_bbox()` and `crop_around_building()`

2. **Normalize resolution**:
   - Maintains aspect ratio during resizing
   - Resizes to maximum dimension of 2048px (configurable)
   - Uses high-quality LANCZOS resampling for best quality
   - Handles both upscaling (if image too small) and downscaling (if image too large)
   - **Function**: `normalize_resolution()`

3. **Contrast Enhancement** (Bonus):
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Converts to LAB color space for better normalization
   - Enhances image quality for better analysis results
   - **Function**: `enhance_contrast()`

**Code Reference**:
```python
# backend/yolo_service/preprocessing.py

# Main preprocessing function
def preprocess_image(image, model, crop_building=True, max_dimension=2048, 
                    apply_contrast_enhancement=True)

# Building detection
def detect_building_bbox(image, model, conf_threshold=0.3)

# Resolution normalization
def normalize_resolution(image, max_dimension=2048, min_dimension=None)

# Smart cropping
def crop_around_building(image, building_bbox, padding_ratio=0.2, 
                        min_padding_pixels=50)

# Contrast enhancement
def enhance_contrast(image, method="clahe")
```

**Workflow**:
1. Convert image to RGB mode (if needed)
2. Normalize resolution (maintain aspect ratio, max 2048px)
3. Detect building using YOLO
4. Crop around detected building (with smart padding)
5. Enhance contrast using CLAHE
6. Return preprocessed image with metadata

**Output**:
- `preprocessed_image`: Processed PIL Image
- `original_size`: Original dimensions
- `preprocessed_size`: Processed dimensions
- `crop_box`: Crop coordinates (x1, y1, x2, y2) or None
- `normalized`: Whether resolution was normalized
- `enhanced`: Whether contrast was enhanced
- `success`: Processing success status

### 2. ROOF SEGMENTATION MODEL
**Status**: ✅ **Implemented** (Computer Vision + Deep Learning)

**Location**: `backend/yolo_service/segmentation.py`, `backend/yolo_service/dl_segmentation.py`

**Current Implementation**:

#### A. Computer Vision Approach (Default):
- **Gaussian Blur**: Noise reduction with 5x5 kernel
- **Canny Edge Detection**: Thresholds (50, 150) for edge detection
- **Contour Detection**: `cv2.findContours()` with `RETR_EXTERNAL` mode
- **Area Filtering**: Removes small noise (minimum 1% of image area)
- **Largest Contour Selection**: Identifies roof as largest detected contour
- **Bounding Rectangle**: Extracts roof bounding box coordinates
- **Methods**: `enhanced_canny`, `watershed`, `contour_based`

#### B. Deep Learning Approach (✅ **NEW - Implemented**):
- **U-Net**: Encoder-decoder architecture with skip connections
  - Encoder: ResNet34 backbone (ImageNet pretrained)
  - Decoder: Upsampling with skip connections
  - Output: Binary roof mask
- **DeepLabv3+**: Atrous convolution for multi-scale feature extraction
  - Encoder: ResNet50 backbone (ImageNet pretrained)
  - ASPP (Atrous Spatial Pyramid Pooling) module
  - Decoder: Refines segmentation boundaries
- **HRNet**: High-Resolution Network maintaining high-resolution representations
  - Multi-resolution parallel branches
  - Repeated multi-scale fusions
  - Preserves spatial details

**Model Loading**:
- Supports pretrained models (ImageNet weights)
- Supports custom trained models (load from file path)
- Automatic fallback to computer vision if PyTorch unavailable
- GPU acceleration when available

**Code Reference**:
```python
# backend/yolo_service/segmentation.py
def segment_roof(image, method="enhanced_canny", dl_model_path=None)
# Methods: 'enhanced_canny', 'watershed', 'contour_based', 'unet', 'deeplabv3plus', 'hrnet'

# backend/yolo_service/dl_segmentation.py
class UNetSegmentation
class DeepLabV3PlusSegmentation
class HRNetSegmentation
def segment_with_dl_model(image, model_type="unet", model_path=None)
```

**Usage**:
```python
# Computer Vision (default)
result = segment_roof(image, method="enhanced_canny")

# Deep Learning Models
result = segment_roof(image, method="unet")
result = segment_roof(image, method="deeplabv3plus")
result = segment_roof(image, method="hrnet")

# With custom trained model
result = segment_roof(image, method="unet", dl_model_path="./models/unet_roof.pth")
```

**Dependencies**:
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Vision models and transforms
- `segmentation-models-pytorch>=0.3.3` - Pre-built segmentation models

### 3. ROOF MASK
**Location**: `backend/yolo_service.py` - `detect_roof_edges()` method

**Implementation**:
- **Edge Image Generation**: Creates visual representation of detected edges
- **Contour Drawing**: Draws roof contour in green (thickness: 2px)
- **Bounding Box**: Draws roof bounding rectangle in red (thickness: 2px)
- **Area Calculation**: Calculates roof area in pixels (`cv2.contourArea()`)
- **Coordinate Extraction**: Returns roof bounding box (x, y, width, height)

**Output**:
- `roof_contour_area_pixels`: Total roof area in pixels
- `roof_bbox`: Bounding box coordinates
- `edge_image`: Visualized edge detection result
- `contours_found`: Number of detected contours

**Code Reference**:
```python
# backend/yolo_service.py (lines 347-357)
largest_contour = max(roof_contours, key=cv2.contourArea)
roof_area_pixels = cv2.contourArea(largest_contour)
x, y, w, h = cv2.boundingRect(largest_contour)
edge_image = img_array.copy()
cv2.drawContours(edge_image, [largest_contour], -1, (0, 255, 0), 2)
cv2.rectangle(edge_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
```

### 4. DETECT OBSTRUCTIONS (YOLOv12)
**Location**: `backend/yolo_service.py` - `detect_objects()`, `identify_obstructions()`

**Implementation**:
- **Model**: YOLOv12 (Ultralytics) - Latest YOLO version
- **Priority Detection**: Enhanced detection for Solar, Human, Tree (lower confidence threshold: 0.15)
- **Standard Detection**: Other objects (confidence threshold: 0.25)
- **Object Categories**:
  - **Priority**: Solar panels, Humans, Trees
  - **Obstructions**: Chimneys, Skylights, Water tanks, HVAC units, Vents, Dishes
  - **Additional**: Vehicles, Animals, Equipment, Fences, Buildings

**Visualization**:
- Color-coded bounding boxes (thickness: 3-6px)
- Priority indicator (⭐) on labels
- Confidence percentages displayed
- Shadow effects for better visibility

**Code Reference**:
```python
# backend/yolo_service.py (lines 216-313)
# Priority keywords for enhanced detection
priority_keywords = ['solar', 'panel', 'photovoltaic', 'person', 'human', 'tree', 'plant']
# Run YOLO with lower confidence for priority objects
results = self.model(image, conf=0.15, verbose=False)
```

### 5. OBSTRUCTION MASKS
**Location**: `backend/yolo_service.py` - `identify_obstructions()`

**Implementation**:
- **Bounding Box Extraction**: Per-object bounding boxes from YOLO
- **Area Calculation**: Pixel area for each obstruction
- **Classification**: Categorizes objects into obstruction types
- **Area Aggregation**: Calculates total obstruction area

**Current Status**: Using bounding boxes (not pixel-level segmentation masks yet)

**Future Enhancement**: Per-object segmentation masks using Mask R-CNN or YOLOv12 segmentation

### 6. SUBTRACT OBSTRUCTIONS
**Location**: `backend/yolo_service.py` - `calculate_usable_roof_area()`

**Implementation**:
- **Total Roof Area**: From roof edge detection (pixels)
- **Obstruction Area**: Sum of all detected obstruction areas (pixels)
- **Usable Area Calculation**: `usable_area = roof_area - obstruction_area`
- **Pixel-to-Meter Conversion**: Estimates meter² using pixel ratio
- **Percentage Calculation**: Usable percentage and obstruction percentage

**Code Reference**:
```python
# backend/yolo_service.py
usable_area_pixels = total_roof_area_pixels - total_obstruction_area_pixels
# Convert pixels to m² (estimated ratio)
usable_area_m2 = usable_area_pixels * pixel_to_meter_ratio
```

### 7. USABLE ROOF AREA POLYGON
**Location**: `backend/yolo_service.py` - `calculate_usable_roof_area()`

**Current Implementation**:
- **Area Calculation**: Calculates usable area in m²
- **Percentage Metrics**: Usable percentage and obstruction percentage
- **Bounding Box**: Returns roof bounding rectangle

**Future Enhancement**:
- Convert mask to polygon (contour to polygon conversion)
- Smooth edges (morphological operations)
- Precise polygon coordinates for layout optimization

### 8. SOLAR LAYOUT OPTIMIZER
**Status**: 🔄 Planned / Future Enhancement

**Planned Features**:
- Automated panel placement algorithm
- Orientation optimization (south-facing preferred)
- Spacing rules (minimum gaps between panels)
- Panel size optimization
- Array layout generation

### 9. SOLAR ENERGY COMPUTATION
**Location**: `backend/main.py` - ROI calculation section

**Implementation**:
- **System Size (kWp)**: Based on usable roof area
- **Annual Energy (kWh/year)**: Estimated generation
- **Efficiency Loss**: From hotspot and dirt detection
- **Shading Factor**: Basic estimation (can be enhanced)

**Code Reference**:
```python
# backend/main.py (lines 148-200)
# Calculate system size based on usable area
system_size_kw = (usable_roof_area * 0.15)  # ~150W per m²
annual_energy_kwh = system_size_kw * 1200  # Average generation factor
```

### 10. FINAL OUTPUT
**Location**: `backend/main.py`, `frontend/src/components/ResultsDisplay.tsx`

**Backend Output**:
- Usable roof area (m²)
- Detected objects list
- Obstruction details
- Solar panel analysis (hotspots, dirt)
- Energy predictions
- ROI estimates
- Annotated image (base64)

**Frontend Display**:
- Interactive results visualization
- Color-coded object detection
- Solar panel health metrics
- Hotspot and dirt accumulation details
- Energy and savings estimates
- Download capabilities

## Additional Features

### Image Quality Optimization
- **Format**: PNG (lossless) for annotated images
- **Resolution**: Maximum 2048px (maintains quality, reasonable file size)
- **Resampling**: LANCZOS algorithm for high-quality resizing
- **Quality**: 95% JPEG quality (when applicable)

### Solar Panel Analysis
- **Hotspot Detection**: Thermal anomalies and damaged cells
- **Dirt Accumulation**: Soiling level detection
- **Panel Health Score**: Overall panel condition (0-100%)
- **Efficiency Loss**: Estimated energy loss percentage

### Database Integration
- **SQLite Database**: Stores analysis results
- **Session Management**: Tracks analysis sessions
- **Result Persistence**: Historical analysis data
- **API Endpoints**: CRUD operations for analyses

## Technology Stack

- **Backend**: FastAPI (Python)
- **Computer Vision**: OpenCV, YOLOv12 (Ultralytics)
- **Image Processing**: Pillow (PIL), NumPy
- **Database**: SQLite (SQLAlchemy ORM)
- **Frontend**: React + Vite + TypeScript
- **3D Visualization**: Three.js
- **Styling**: Tailwind CSS

## File Structure

```
backend/
├── main.py              # FastAPI endpoints
├── database.py          # Database connection
├── models.py            # SQLAlchemy models
├── requirements.txt     # Python dependencies
├── solar_analysis.db    # SQLite database
├── models/
│   ├── yolo12n.pt      # YOLOv12 model
│   └── yolov8n.pt      # YOLOv8 model (fallback)
└── yolo_service/
    ├── __init__.py      # Service exports
    ├── service.py       # Main YOLO service
    ├── preprocessing.py # Image preprocessing
    ├── segmentation.py  # Roof segmentation (CV + DL)
    ├── dl_segmentation.py # Deep learning models (U-Net, DeepLabv3+, HRNet)
    ├── detection.py     # Object detection
    ├── analysis.py      # Hotspot & dirt analysis
    └── calculations.py # Area calculations

frontend/
├── src/
│   ├── components/
│   │   ├── Header.tsx           # Navigation
│   │   ├── Hero/
│   │   │   ├── index.tsx        # Hero orchestrator
│   │   │   ├── FirstPage.tsx    # Landing (Black Hole)
│   │   │   ├── SecondPage.tsx   # Solar system
│   │   │   ├── ThirdPage.tsx    # 3D city
│   │   │   └── BlackHole.tsx    # Three.js black hole
│   │   ├── Features.tsx         # Features showcase
│   │   ├── AnalysisForm.tsx     # Image upload form
│   │   ├── ResultsDisplay.tsx   # Results visualization
│   │   ├── LoadingProgress.tsx   # Loading indicator
│   │   ├── Toast.tsx            # Notifications
│   │   ├── ImagePreview.tsx     # Image preview
│   │   └── Footer.tsx           # Footer
│   ├── services/
│   │   └── api.ts               # API client (axios)
│   ├── App.tsx                  # Main app component
│   ├── main.tsx                 # Entry point
│   └── index.css                # Global styles
├── package.json                 # Dependencies
└── vite.config.ts               # Vite configuration
```

## Frontend Implementation

### Frontend Architecture

**Location**: `frontend/src/`

**Technology Stack**:
- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite (fast HMR)
- **Styling**: Tailwind CSS with custom liquid glass effects
- **HTTP Client**: Axios
- **3D Graphics**: Three.js
- **State Management**: React Hooks (useState, useEffect, useRef)

### Frontend Components

#### 1. **App.tsx** - Main Application
- Orchestrates all components
- Manages global state (results, loading, toast)
- Handles analysis workflow
- Black theme background

#### 2. **AnalysisForm.tsx** - Image Upload
**Features**:
- Drag & drop file upload
- File validation (type, size)
- Multiple file selection
- Image preview
- Error handling
- Liquid glass UI

**Code Reference**:
```typescript
// frontend/src/components/AnalysisForm.tsx
const handleFileChange = (selectedFiles: FileList | null)
const handleSubmit = async (e: React.FormEvent)
const handleDragOver = (e: React.DragEvent)
```

#### 3. **ResultsDisplay.tsx** - Results Display
**Current Features**:
- Preprocessing status display
- Fade-in animations
- Liquid glass styling

**Future** (when re-enabled):
- Roof segmentation visualization
- Roof mask display
- Object detection results
- Metrics and statistics

#### 4. **Hero Components** - Landing Pages
- **FirstPage.tsx**: Black hole background with CTA
- **SecondPage.tsx**: Solar system animation with features
- **ThirdPage.tsx**: 3D city scene with "How It Works"
- **BlackHole.tsx**: Three.js black hole visualization

#### 5. **API Service** - Backend Communication
**Location**: `frontend/src/services/api.ts`

**Implementation**:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export const analyzeRooftop = async (files: File[])
```

**Features**:
- Configurable API URL
- FormData multipart upload
- 60-second timeout
- Error handling

### Frontend Workflow

1. **User Uploads Images**:
   - Drag & drop or file picker
   - Validation (type, size)
   - Preview display

2. **API Request**:
   - FormData creation
   - POST to `/api/analyze`
   - Loading state shown

3. **Backend Processing**:
   - Image preprocessing
   - Roof segmentation
   - Object detection
   - Analysis computation

4. **Results Display**:
   - Results received
   - Displayed in `ResultsDisplay`
   - Smooth scroll to results

### UI/UX Features

**Liquid Glass Effects**:
- `.liquid-glass`: Standard glassmorphism
- `.liquid-glass-strong`: Stronger effect
- `.liquid-glass-hover`: Hover state
- Backdrop blur with transparency
- Gradient borders

**3D Visualizations**:
- **Black Hole**: Three.js with accretion disk, gravitational lensing
- **Solar System**: CSS animations with planets
- **3D City**: Three.js city scene with buildings

**Responsive Design**:
- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px)
- Full-screen Hero sections
- Adaptive layouts

**Theme**:
- Pure black background (`#000000`)
- White text (`#ffffff`)
- No light mode (black-only theme)
- Gradient accents (blue, purple, green)

### Environment Configuration

**Environment Variables**:
- `VITE_API_URL`: Backend API URL (default: `http://localhost:8000`)

**Build**:
- Development: `npm run dev` (Vite dev server)
- Production: `npm run build` (optimized build)
- Preview: `npm run preview` (preview production build)
