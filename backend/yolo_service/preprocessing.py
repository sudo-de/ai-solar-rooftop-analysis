"""
Image Preprocessing Module
Handles image normalization, cropping, and enhancement

Features:
- Crop around building: Automatically detects and crops to building area
- Normalize resolution: Resizes to optimal size while maintaining aspect ratio
- Contrast enhancement: Improves image quality for better analysis
"""

from typing import Dict, Tuple, Optional, List
from PIL import Image
import numpy as np
import cv2


def detect_building_bbox(image: Image.Image, model, conf_threshold: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect building bounding box using YOLO model
    
    Args:
        image: PIL Image object
        model: YOLO model for building detection
        conf_threshold: Confidence threshold for detection
    
    Returns:
        Tuple of (x1, y1, x2, y2) bounding box coordinates, or None if not found
    """
    try:
        img_array = np.array(image)
        results = model(img_array, conf=conf_threshold, classes=[], verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # Building-related keywords
            building_keywords = [
                'building', 'house', 'roof', 'structure', 'construction',
                'residential', 'commercial', 'edifice', 'architecture'
            ]
            
            # Find all building detections
            building_detections = []
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id].lower()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Check if it's a building/roof/structure
                if any(keyword in class_name for keyword in building_keywords):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    building_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'area': area,
                        'confidence': confidence,
                        'class': class_name
                    })
            
            if not building_detections:
                return None
            
            # Strategy 1: Use largest building detection
            largest = max(building_detections, key=lambda x: x['area'])
            
            # Strategy 2: If multiple buildings, try to merge nearby ones
            if len(building_detections) > 1:
                # Check if buildings are close together (within 20% of image size)
                img_width, img_height = image.size
                max_distance = min(img_width, img_height) * 0.2
                
                merged_bboxes = [largest['bbox']]
                for det in building_detections:
                    if det['bbox'] == largest['bbox']:
                        continue
                    
                    # Calculate distance between bounding boxes
                    x1_l, y1_l, x2_l, y2_l = largest['bbox']
                    x1_d, y1_d, x2_d, y2_d = det['bbox']
                    
                    # Center points
                    center_l = ((x1_l + x2_l) / 2, (y1_l + y2_l) / 2)
                    center_d = ((x1_d + x2_d) / 2, (y1_d + y2_d) / 2)
                    
                    distance = np.sqrt((center_l[0] - center_d[0])**2 + (center_l[1] - center_d[1])**2)
                    
                    if distance < max_distance:
                        # Merge bounding boxes
                        merged_x1 = min(x1_l, x1_d)
                        merged_y1 = min(y1_l, y1_d)
                        merged_x2 = max(x2_l, x2_d)
                        merged_y2 = max(y2_l, y2_d)
                        merged_bboxes.append((merged_x1, merged_y1, merged_x2, merged_y2))
                
                # Use the merged bounding box that covers all nearby buildings
                if len(merged_bboxes) > 1:
                    all_x1 = [b[0] for b in merged_bboxes]
                    all_y1 = [b[1] for b in merged_bboxes]
                    all_x2 = [b[2] for b in merged_bboxes]
                    all_y2 = [b[3] for b in merged_bboxes]
                    return (int(min(all_x1)), int(min(all_y1)), int(max(all_x2)), int(max(all_y2)))
            
            return largest['bbox']
        
        return None
    except Exception as e:
        print(f"Building detection error: {str(e)}")
        return None


def normalize_resolution(image: Image.Image, max_dimension: int = 2048, 
                        min_dimension: Optional[int] = None) -> Image.Image:
    """
    Normalize image resolution while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_dimension: Maximum dimension (width or height) in pixels
        min_dimension: Minimum dimension (optional, for upscaling small images)
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    # Calculate scaling ratio
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        print(f"📐 Resolution normalized: {width}x{height} → {new_width}x{new_height} (ratio: {ratio:.3f})")
    elif min_dimension and (width < min_dimension or height < min_dimension):
        # Upscale small images if needed
        ratio = max(min_dimension / width, min_dimension / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        print(f"📐 Resolution upscaled: {width}x{height} → {new_width}x{new_height} (ratio: {ratio:.3f})")
    else:
        return image  # No resizing needed
    
    # Use high-quality resampling
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def crop_around_building(image: Image.Image, building_bbox: Tuple[int, int, int, int],
                        padding_ratio: float = 0.2, min_padding_pixels: int = 50) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crop image around detected building with smart padding
    
    Args:
        image: PIL Image object
        building_bbox: Building bounding box (x1, y1, x2, y2)
        padding_ratio: Padding as ratio of building size (default: 20%)
        min_padding_pixels: Minimum padding in pixels
    
    Returns:
        Tuple of (cropped_image, crop_box)
    """
    x1, y1, x2, y2 = building_bbox
    width, height = image.size
    
    # Calculate building dimensions
    building_width = x2 - x1
    building_height = y2 - y1
    
    # Calculate padding (use larger of ratio-based or minimum)
    padding_x = max(int(building_width * padding_ratio), min_padding_pixels)
    padding_y = max(int(building_height * padding_ratio), min_padding_pixels)
    
    # Calculate crop box with padding, ensuring it stays within image bounds
    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)
    crop_x2 = min(width, x2 + padding_x)
    crop_y2 = min(height, y2 + padding_y)
    
    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
    cropped_image = image.crop(crop_box)
    
    print(f"✂️  Cropped around building: {crop_box} (padding: {padding_x}x{padding_y}px)")
    print(f"   Building size: {building_width}x{building_height}px, Crop size: {crop_x2-crop_x1}x{crop_y2-crop_y1}px")
    
    return cropped_image, crop_box


def enhance_contrast(image: Image.Image, method: str = "clahe") -> Image.Image:
    """
    Enhance image contrast for better analysis
    
    Args:
        image: PIL Image object
        method: Enhancement method ('clahe' or 'histogram')
    
    Returns:
        Enhanced PIL Image
    """
    img_array = np.array(image)
    
    if method == "clahe":
        # Convert to LAB color space for better normalization
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back to RGB
        lab = cv2.merge([l, a, b])
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Histogram equalization (fallback)
        enhanced_img = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(enhanced_img)


def preprocess_image(image: Image.Image, model, target_size: Optional[Tuple[int, int]] = None, 
                    crop_building: bool = True, max_dimension: int = 2048,
                    apply_contrast_enhancement: bool = True) -> Dict:
    """
    Preprocess image for roof analysis
    
    This function performs:
    1. Resolution normalization (maintains aspect ratio, max 2048px)
    2. Building detection and cropping (auto-crop around detected building)
    3. Contrast enhancement (CLAHE for better analysis)
    
    Args:
        image: PIL Image object
        model: YOLO model for building detection
        target_size: Target resolution (width, height). If None, maintains aspect ratio with max_dimension
        crop_building: Whether to auto-crop around detected building
        max_dimension: Maximum dimension for resolution normalization (default: 2048px)
        apply_contrast_enhancement: Whether to enhance contrast using CLAHE
    
    Returns:
        Dictionary with preprocessed image and metadata:
        - preprocessed_image: Processed PIL Image
        - original_size: Original image dimensions (width, height)
        - preprocessed_size: Processed image dimensions (width, height)
        - original_mode: Original image color mode
        - crop_box: Crop box coordinates (x1, y1, x2, y2) or None
        - normalized: Whether resolution was normalized
        - enhanced: Whether contrast was enhanced
        - success: Whether preprocessing succeeded
    """
    try:
        original_size = image.size
        original_mode = image.mode
        
        print(f"🖼️  Starting preprocessing: {original_size[0]}x{original_size[1]}px, mode: {original_mode}")
        
        # Step 1: Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"   Converted to RGB mode")
        
        # Step 2: Normalize resolution
        if target_size is None:
            image = normalize_resolution(image, max_dimension=max_dimension)
        else:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            print(f"📐 Resized to target size: {target_size[0]}x{target_size[1]}px")
        
        # Step 3: Auto-crop around building if requested
        crop_box = None
        if crop_building:
            building_bbox = detect_building_bbox(image, model, conf_threshold=0.3)
            
            if building_bbox:
                image, crop_box = crop_around_building(image, building_bbox, padding_ratio=0.2)
            else:
                print("⚠️  No building detected, using full image")
        
        # Step 4: Enhance contrast
        if apply_contrast_enhancement:
            image = enhance_contrast(image, method="clahe")
            print("✨ Contrast enhanced using CLAHE")
        
        return {
            "preprocessed_image": image,
            "original_size": original_size,
            "preprocessed_size": image.size,
            "original_mode": original_mode,
            "crop_box": crop_box,
            "normalized": True,
            "enhanced": apply_contrast_enhancement,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Image preprocessing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "preprocessed_image": image,
            "original_size": image.size,
            "preprocessed_size": image.size,
            "original_mode": image.mode,
            "crop_box": None,
            "normalized": False,
            "enhanced": False,
            "success": False,
            "error": str(e)
        }
        
        # Normalize image (enhance contrast and brightness)
        img_array = np.array(image)
        
        # Convert to LAB color space for better normalization
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back to RGB
        lab = cv2.merge([l, a, b])
        normalized_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image
        preprocessed_image = Image.fromarray(normalized_img)
        
        return {
            "preprocessed_image": preprocessed_image,
            "original_size": original_size,
            "preprocessed_size": preprocessed_image.size,
            "original_mode": original_mode,
            "crop_box": crop_box,
            "normalized": True,
            "success": True
        }
        
    except Exception as e:
        print(f"Image preprocessing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "preprocessed_image": image,
            "original_size": image.size,
            "preprocessed_size": image.size,
            "original_mode": image.mode,
            "crop_box": None,
            "normalized": False,
            "success": False,
            "error": str(e)
        }