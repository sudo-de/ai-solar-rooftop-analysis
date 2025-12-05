"""
Object Detection Module
Handles YOLO object detection and visualization
"""

from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64


def detect_objects(image: Image.Image, model, conf_threshold: float = 0.25) -> Dict:
    """
    Detect objects in rooftop image
    
    Args:
        image: PIL Image object
        model: YOLO model instance
        conf_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary with detection results
    """
    try:
        # Run YOLO inference with lower confidence to catch more objects
        results = model(image, conf=0.15, verbose=False)  # Lower confidence to catch more
        
        # Process results with priority for Solar, Human, Tree
        detections = []
        detection_set = set()  # Track unique detections to avoid duplicates
        
        # Priority keywords for enhanced detection (Solar, Human, Tree)
        priority_keywords = ['solar', 'panel', 'photovoltaic', 'pv', 'array', 'module',
                           'person', 'human', 'people', 'man', 'woman', 'child',
                           'tree', 'plant', 'vegetation', 'bush', 'shrub', 'foliage']
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                class_name_lower = class_name.lower()
                
                # Check if it's a priority object
                is_priority = any(keyword in class_name_lower for keyword in priority_keywords)
                
                # Filter: keep priority objects even with lower confidence, or standard objects above threshold
                if is_priority or confidence >= conf_threshold:
                    # Create unique key for detection
                    det_key = (int(x1/10), int(y1/10), int(x2/10), int(y2/10), class_id)  # Rounded to avoid exact duplicates
                    
                    if det_key not in detection_set:
                        detection_set.add(det_key)
                        detections.append({
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": round(confidence, 3),
                            "priority": is_priority,
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1)
                            }
                        })
        
        # Sort detections: priority objects first, then by confidence
        detections.sort(key=lambda x: (not x.get("priority", False), -x["confidence"]))
        
        annotated_image = image.copy()
        
        if detections:
            # Create custom annotated image with ENHANCED clear visualization
            img_array = np.array(image)
            # Ensure image is RGB
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            annotated_image = img_array.copy()
            
            # Enhanced color mapping for better visibility - Prioritizing Solar, Human, Tree
            def get_color_for_class(class_name: str) -> Tuple:
                """Get distinct color for each object type with priority for Solar, Human, Tree"""
                class_lower = class_name.lower()
                
                # SOLAR PANELS - Highest Priority (Gold, very thick)
                if any(word in class_lower for word in ['solar', 'panel', 'photovoltaic', 'pv', 'array', 'module']):
                    return (255, 215, 0), 6  # Gold, very thick
                
                # HUMANS - High Priority (Bright Pink, thick)
                elif any(word in class_lower for word in ['person', 'human', 'people', 'man', 'woman', 'child', 'boy', 'girl']):
                    return (255, 20, 147), 5  # Deep Pink, thick
                
                # TREES - High Priority (Forest Green, thick)
                elif any(word in class_lower for word in ['tree', 'plant', 'vegetation', 'bush', 'shrub', 'foliage', 'branch']):
                    return (34, 139, 34), 5  # Forest Green, thick
                
                # Buildings and structures
                elif any(word in class_lower for word in ['building', 'house', 'roof', 'structure', 'construction']):
                    return (0, 255, 0), 4  # Bright Green, medium
                
                # Obstructions (rooftop equipment)
                elif any(word in class_lower for word in ['chimney', 'hvac', 'tank', 'skylight', 'vent', 'ventilation']):
                    return (0, 0, 255), 5  # Bright Red, thick
                
                # Vehicles
                elif any(word in class_lower for word in ['car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bike', 'bicycle']):
                    return (255, 140, 0), 4  # Dark Orange, medium
                
                # Animals (birds, pets, wildlife)
                elif any(word in class_lower for word in ['bird', 'dog', 'cat', 'animal', 'pet', 'squirrel', 'raccoon']):
                    return (138, 43, 226), 4  # Blue Violet, medium
                
                # Equipment and tools
                elif any(word in class_lower for word in ['ladder', 'tool', 'equipment', 'machinery', 'generator']):
                    return (255, 69, 0), 4  # Red Orange, medium
                
                # Satellite and antennas
                elif any(word in class_lower for word in ['satellite', 'dish', 'antenna', 'tower']):
                    return (0, 191, 255), 4  # Deep Sky Blue, medium
                
                # Water features
                elif any(word in class_lower for word in ['pool', 'pond', 'fountain', 'water']):
                    return (0, 206, 209), 4  # Dark Turquoise, medium
                
                # Fences and barriers
                elif any(word in class_lower for word in ['fence', 'barrier', 'railing', 'gate']):
                    return (128, 128, 128), 3  # Gray, thin
                
                # Other objects
                else:
                    return (255, 165, 0), 3  # Orange, thin
            
            # Draw bounding boxes with enhanced labels
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                class_name = det["class"]
                confidence = det["confidence"]
                
                # Get color and thickness
                color, thickness = get_color_for_class(class_name)
                
                # Draw thicker bounding box with shadow effect for visibility
                # Shadow
                cv2.rectangle(annotated_image, (x1+2, y1+2), (x2+2, y2+2), (0, 0, 0), thickness+2)
                # Main box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare enhanced label text with priority indicator
                priority_indicator = "⭐ " if det.get("priority", False) else ""
                label = f"{priority_indicator}{class_name.upper()} {confidence:.1%}"
                
                # Calculate text size with larger font
                font = cv2.FONT_HERSHEY_BOLD
                font_scale = max(0.8, min(1.2, (x2 - x1) / 200))  # Scale based on box size
                text_thickness = max(2, int(font_scale * 2))
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                # Draw label background with padding
                padding = 5
                label_y = max(y1 - 10, text_height + 15)
                label_x = x1
                
                # Background rectangle with border
                cv2.rectangle(annotated_image, 
                             (label_x - padding, label_y - text_height - padding),
                             (label_x + text_width + padding, label_y + baseline + padding),
                             (0, 0, 0), -1)  # Black background
                cv2.rectangle(annotated_image, 
                             (label_x - padding, label_y - text_height - padding),
                             (label_x + text_width + padding, label_y + baseline + padding),
                             color, 2)  # Colored border
                
                # Draw label text with shadow for readability
                cv2.putText(annotated_image, label,
                          (label_x, label_y),
                          font, font_scale, (0, 0, 0), text_thickness + 1, cv2.LINE_AA)  # Shadow
                cv2.putText(annotated_image, label,
                          (label_x, label_y),
                          font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)  # White text
            
            annotated_image = Image.fromarray(annotated_image)
        else:
            # No detections, return original image
            annotated_image = image.copy()
        
        # Convert annotated image to base64 with MAXIMUM quality and resolution
        buffered = BytesIO()
        # Resize if too large to maintain quality while keeping file size reasonable
        max_dimension = 2048
        if annotated_image.width > max_dimension or annotated_image.height > max_dimension:
            ratio = min(max_dimension / annotated_image.width, max_dimension / annotated_image.height)
            new_width = int(annotated_image.width * ratio)
            new_height = int(annotated_image.height * ratio)
            annotated_image = annotated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save as PNG for best quality (no compression artifacts)
        annotated_image.save(buffered, format="PNG", optimize=False)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        annotated_image_base64 = f"data:image/png;base64,{img_str}"
        
        return {
            "detections": detections,
            "total_detections": len(detections),
            "annotated_image_base64": annotated_image_base64,
            "model": str(model)
        }
        
    except Exception as e:
        print(f"YOLO detection error: {str(e)}")
        return {
            "detections": [],
            "total_detections": 0,
            "annotated_image_base64": None,
            "error": str(e)
        }


def identify_obstructions(detections: List[Dict]) -> Dict:
    """
    Identify and classify rooftop obstructions
    
    Args:
        detections: List of detected objects
    
    Returns:
        Dictionary with classified obstructions
    """
    # Enhanced obstruction categories - including more object types
    obstruction_keywords = {
        "chimney": ["chimney", "smokestack", "vent"],
        "skylight": ["skylight", "window", "glass", "dome"],
        "water_tank": ["tank", "container", "barrel", "cistern"],
        "hvac": ["hvac", "air conditioner", "ac unit", "ventilation", "fan"],
        "satellite_dish": ["satellite", "dish", "antenna", "tower"],
        "solar_panel": ["solar", "panel", "photovoltaic", "pv", "array", "module"],
        "tree": ["tree", "branch", "plant", "vegetation", "bush", "shrub", "foliage"],
        "human": ["person", "human", "people", "man", "woman", "child"],
        "vehicle": ["car", "vehicle", "truck", "bus", "motorcycle", "bike", "bicycle"],
        "animal": ["bird", "dog", "cat", "animal", "pet", "squirrel", "raccoon"],
        "equipment": ["ladder", "tool", "equipment", "machinery", "generator"],
        "fence": ["fence", "barrier", "railing", "gate"],
        "other": []
    }
    
    obstructions = {
        "chimneys": [],
        "skylights": [],
        "water_tanks": [],
        "hvac_units": [],
        "satellite_dishes": [],
        "solar_panels": [],
        "trees": [],
        "humans": [],
        "vehicles": [],
        "animals": [],
        "equipment": [],
        "fences": [],
        "other": []
    }
    
    total_obstruction_area_pixels = 0
    
    for det in detections:
        class_name = det["class"].lower()
        bbox = det["bbox"]
        area_pixels = bbox["width"] * bbox["height"]
        
        obstruction_info = {
            "class": det["class"],
            "confidence": det["confidence"],
            "bbox": bbox,
            "area_pixels": area_pixels
        }
        
        # Classify obstruction
        classified = False
        for category, keywords in obstruction_keywords.items():
            if any(keyword in class_name for keyword in keywords):
                if category == "chimney":
                    obstructions["chimneys"].append(obstruction_info)
                elif category == "skylight":
                    obstructions["skylights"].append(obstruction_info)
                elif category == "water_tank":
                    obstructions["water_tanks"].append(obstruction_info)
                elif category == "hvac":
                    obstructions["hvac_units"].append(obstruction_info)
                elif category == "satellite_dish":
                    obstructions["satellite_dishes"].append(obstruction_info)
                elif category == "solar_panel":
                    obstructions["solar_panels"].append(obstruction_info)
                elif category == "tree":
                    obstructions["trees"].append(obstruction_info)
                elif category == "human":
                    obstructions["humans"].append(obstruction_info)
                elif category == "vehicle":
                    obstructions["vehicles"].append(obstruction_info)
                elif category == "animal":
                    obstructions["animals"].append(obstruction_info)
                elif category == "equipment":
                    obstructions["equipment"].append(obstruction_info)
                elif category == "fence":
                    obstructions["fences"].append(obstruction_info)
                else:
                    obstructions["other"].append(obstruction_info)
                
                total_obstruction_area_pixels += area_pixels
                classified = True
                break
        
        if not classified and det["class"].lower() not in ["building", "house", "roof"]:
            obstructions["other"].append(obstruction_info)
            total_obstruction_area_pixels += area_pixels
    
    return {
        "obstructions": obstructions,
        "total_obstruction_area_pixels": total_obstruction_area_pixels,
        "total_obstruction_count": sum(len(v) for v in obstructions.values())
    }


def detect_solar_panels(image: Image.Image, detections: List[Dict], model) -> List[Dict]:
    """
    Detect solar panels using YOLOv12 with enhanced detection for photovoltaics
    
    Args:
        image: PIL Image object
        detections: List of detected objects from YOLO
        model: YOLO model instance
    
    Returns:
        List of detected solar panels with locations
    """
    solar_panels = []
    
    # Filter for solar panel detections from YOLOv12
    for det in detections:
        class_name = det.get("class", "").lower()
        # YOLOv12 can detect various solar-related objects
        if any(keyword in class_name for keyword in ["solar", "panel", "photovoltaic", "pv", "array"]):
            solar_panels.append(det)
    
    # Enhanced detection using YOLOv12 with lower confidence for solar panels
    if len(solar_panels) == 0:
        try:
            # Run YOLOv12 with lower confidence specifically for solar panels
            results = model(image, conf=0.15, classes=[], verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id].lower()
                    
                    # Check if it's a rectangular structure that could be a solar panel
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0
                    area = width * height
                    
                    # Solar panels are typically rectangular structures
                    # Also check for building/roof structures that might contain panels
                    if (any(keyword in class_name for keyword in ["solar", "panel", "photovoltaic", "building", "roof", "structure"]) or
                        (area > 2000 and 0.3 < aspect_ratio < 3.0 and confidence > 0.2)):
                        solar_panels.append({
                            "class": "solar_panel" if "solar" in class_name or "panel" in class_name else f"{model.names[class_id]}_potential_panel",
                            "class_id": class_id,
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(width),
                                "height": float(height)
                            },
                            "area_pixels": float(area)
                        })
        except Exception as e:
            print(f"Error in enhanced solar panel detection: {str(e)}")
    
    # If still no panels, use computer vision fallback
    if not solar_panels:
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            
            # Enhanced edge detection for solar panels
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Morphological operations to connect panel edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Solar panels are typically rectangular
                    if 0.4 < aspect_ratio < 2.5:
                        solar_panels.append({
                            "class": "solar_panel",
                            "confidence": 0.4,
                            "bbox": {
                                "x1": float(x),
                                "y1": float(y),
                                "x2": float(x + w),
                                "y2": float(y + h),
                                "width": float(w),
                                "height": float(h)
                            },
                            "area_pixels": area
                        })
        except Exception as e:
            print(f"Error in fallback solar panel detection: {str(e)}")
    
    return solar_panels

