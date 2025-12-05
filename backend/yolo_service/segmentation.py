"""
Roof Segmentation Module
Handles roof edge detection, segmentation, and mask generation
Supports both computer vision and deep learning methods
Includes U-Net, DeepLabv3+, and HRNet implementations
"""

from typing import Dict, Optional
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64
import os

# Try to import PyTorch and segmentation models
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Deep learning segmentation will use fallback methods.")

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("⚠️  segmentation_models_pytorch not available. Will use torchvision models.")


# ============================================================================
# Deep Learning Segmentation Classes
# ============================================================================

class UNetSegmentation:
    """U-Net model for roof segmentation"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.model_path = model_path
        
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load or create U-Net model"""
        try:
            if SMP_AVAILABLE:
                # Use segmentation_models_pytorch for better U-Net implementation
                self.model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation=None
                )
            else:
                # Fallback: Simple U-Net from scratch
                self.model = self._create_simple_unet()
            
            if self.model_path and os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"✅ Loaded U-Net model from {self.model_path}")
            else:
                print("⚠️  Using untrained U-Net model (will use fallback segmentation)")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading U-Net model: {str(e)}")
            self.model = None
    
    def _create_simple_unet(self):
        """Create a simple U-Net architecture"""
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = self._conv_block(3, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)
                
                # Decoder
                self.dec1 = self._conv_block(512 + 256, 256)
                self.dec2 = self._conv_block(256 + 128, 128)
                self.dec3 = self._conv_block(128 + 64, 64)
                self.final = nn.Conv2d(64, 1, kernel_size=1)
                
            def _conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(nn.MaxPool2d(2)(e1))
                e3 = self.enc3(nn.MaxPool2d(2)(e2))
                e4 = self.enc4(nn.MaxPool2d(2)(e3))
                
                # Decoder
                d1 = nn.functional.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
                d1 = torch.cat([d1, e3], dim=1)
                d1 = self.dec1(d1)
                
                d2 = nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                
                d3 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
                d3 = torch.cat([d3, e1], dim=1)
                d3 = self.dec3(d3)
                
                return torch.sigmoid(self.final(d3))
        
        return SimpleUNet()
    
    def segment(self, image: Image.Image) -> np.ndarray:
        """Perform roof segmentation using U-Net"""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
                mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            # Resize mask back to original size
            mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
            
            return mask
        except Exception as e:
            print(f"U-Net segmentation error: {str(e)}")
            return None


class DeepLabV3PlusSegmentation:
    """DeepLabv3+ model for roof segmentation"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.model_path = model_path
        
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load or create DeepLabv3+ model"""
        try:
            if SMP_AVAILABLE:
                # Use segmentation_models_pytorch for DeepLabv3+
                self.model = smp.DeepLabV3Plus(
                    encoder_name="resnet50",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation=None
                )
            else:
                # Use torchvision DeepLabv3
                self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
                # Modify for binary segmentation
                self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
            
            if self.model_path and os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"✅ Loaded DeepLabv3+ model from {self.model_path}")
            else:
                print("⚠️  Using pretrained DeepLabv3+ model (general purpose, not roof-specific)")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading DeepLabv3+ model: {str(e)}")
            self.model = None
    
    def segment(self, image: Image.Image) -> np.ndarray:
        """Perform roof segmentation using DeepLabv3+"""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                if SMP_AVAILABLE:
                    output = self.model(img_tensor)
                    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
                else:
                    output = self.model(img_tensor)['out']
                    mask = (output[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            # Resize mask back to original size
            mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
            
            return mask
        except Exception as e:
            print(f"DeepLabv3+ segmentation error: {str(e)}")
            return None


class HRNetSegmentation:
    """HRNet (High-Resolution Network) model for roof segmentation"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.model_path = model_path
        
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load or create HRNet model"""
        try:
            if SMP_AVAILABLE:
                # Use segmentation_models_pytorch for HRNet
                self.model = smp.FPN(
                    encoder_name="timm-hrnet_w18",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation=None
                )
            else:
                print("⚠️  HRNet requires segmentation_models_pytorch. Using fallback.")
                self.model = None
                return
            
            if self.model_path and os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"✅ Loaded HRNet model from {self.model_path}")
            else:
                print("⚠️  Using pretrained HRNet model (general purpose, not roof-specific)")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading HRNet model: {str(e)}")
            self.model = None
    
    def segment(self, image: Image.Image) -> np.ndarray:
        """Perform roof segmentation using HRNet"""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
                mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            # Resize mask back to original size
            mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
            
            return mask
        except Exception as e:
            print(f"HRNet segmentation error: {str(e)}")
            return None


def segment_with_dl_model(image: Image.Image, model_type: str = "unet", 
                          model_path: Optional[str] = None) -> Dict:
    """
    Perform roof segmentation using deep learning models
    
    Args:
        image: PIL Image object
        model_type: Model type ('unet', 'deeplabv3plus', 'hrnet')
        model_path: Path to trained model weights (optional)
    
    Returns:
        Dictionary with segmentation results
    """
    if not TORCH_AVAILABLE:
        return {
            "roof_mask": None,
            "success": False,
            "error": "PyTorch not available. Install torch and torchvision."
        }
    
    try:
        # Initialize model
        if model_type.lower() == "unet":
            segmenter = UNetSegmentation(model_path)
        elif model_type.lower() in ["deeplabv3plus", "deeplabv3+", "deeplab"]:
            segmenter = DeepLabV3PlusSegmentation(model_path)
        elif model_type.lower() == "hrnet":
            segmenter = HRNetSegmentation(model_path)
        else:
            return {
                "roof_mask": None,
                "success": False,
                "error": f"Unknown model type: {model_type}"
            }
        
        # Perform segmentation
        mask = segmenter.segment(image)
        
        if mask is None:
            return {
                "roof_mask": None,
                "success": False,
                "error": "Segmentation failed"
            }
        
        # Find contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "roof_mask": mask,
                "roof_contour": None,
                "roof_area_pixels": 0.0,
                "roof_bbox": None,
                "segmented_image": np.array(image),
                "contours_found": 0,
                "method": model_type,
                "success": False
            }
        
        # Find largest contour (roof)
        largest_contour = max(contours, key=cv2.contourArea)
        roof_area_pixels = cv2.contourArea(largest_contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Create visualization
        img_array = np.array(image)
        segmented_image = img_array.copy()
        overlay = segmented_image.copy()
        overlay[mask == 255] = [0, 255, 0]  # Green overlay
        segmented_image = cv2.addWeighted(segmented_image, 0.7, overlay, 0.3, 0)
        
        # Draw contour
        cv2.drawContours(segmented_image, [largest_contour], -1, (0, 255, 0), 3)
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return {
            "roof_mask": mask,
            "roof_contour": largest_contour,
            "roof_area_pixels": float(roof_area_pixels),
            "roof_bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "segmented_image": segmented_image,
            "contours_found": len(contours),
            "method": model_type,
            "success": True
        }
        
    except Exception as e:
        print(f"Deep learning segmentation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "roof_mask": None,
            "roof_contour": None,
            "roof_area_pixels": 0.0,
            "roof_bbox": None,
            "segmented_image": np.array(image),
            "contours_found": 0,
            "method": model_type,
            "success": False,
            "error": str(e)
        }


# ============================================================================
# Main Segmentation Functions
# ============================================================================

def segment_roof(image: Image.Image, method: str = "enhanced_canny", 
                 dl_model_path: Optional[str] = None) -> Dict:
    """
    Enhanced roof segmentation using multiple techniques
    
    Args:
        image: PIL Image object (preprocessed)
        method: Segmentation method:
            - Computer Vision: 'enhanced_canny', 'watershed', 'contour_based'
            - Deep Learning: 'unet', 'deeplabv3plus', 'hrnet'
        dl_model_path: Path to trained deep learning model weights (optional)
    
    Returns:
        Dictionary with roof segmentation results and mask
    """
    # Check if deep learning method is requested
    dl_methods = ['unet', 'deeplabv3plus', 'deeplabv3+', 'deeplab', 'hrnet']
    if method.lower() in dl_methods:
        if TORCH_AVAILABLE:
            print(f"Using deep learning segmentation: {method}")
            return segment_with_dl_model(image, method.lower(), dl_model_path)
        else:
            print(f"⚠️  Deep learning not available, falling back to enhanced_canny")
            method = "enhanced_canny"
    
    try:
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if method == "enhanced_canny":
            # Enhanced Canny edge detection with multiple scales
            # Multi-scale edge detection
            blurred1 = cv2.GaussianBlur(gray, (3, 3), 0)
            blurred2 = cv2.GaussianBlur(gray, (5, 5), 0)
            blurred3 = cv2.GaussianBlur(gray, (7, 7), 0)
            
            edges1 = cv2.Canny(blurred1, 30, 100)
            edges2 = cv2.Canny(blurred2, 50, 150)
            edges3 = cv2.Canny(blurred3, 70, 200)
            
            # Combine edges
            edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
            
            # Morphological operations to connect edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
        elif method == "watershed":
            # Watershed segmentation
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Find sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            img_color = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            markers = cv2.watershed(img_color, markers)
            edges = (markers == -1).astype(np.uint8) * 255
            
        else:  # contour_based
            # Contour-based segmentation
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = (img_array.shape[0] * img_array.shape[1]) * 0.01  # 1% of image
        roof_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Find largest contour (roof)
        if roof_contours:
            largest_contour = max(roof_contours, key=cv2.contourArea)
            roof_area_pixels = cv2.contourArea(largest_contour)
            
            # Create roof mask
            roof_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(roof_mask, [largest_contour], 255)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Smooth mask edges
            kernel = np.ones((5, 5), np.uint8)
            roof_mask = cv2.morphologyEx(roof_mask, cv2.MORPH_CLOSE, kernel)
            roof_mask = cv2.GaussianBlur(roof_mask, (5, 5), 0)
            _, roof_mask = cv2.threshold(roof_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Create visualization
            segmented_image = img_array.copy()
            overlay = segmented_image.copy()
            overlay[roof_mask == 255] = [0, 255, 0]  # Green overlay
            segmented_image = cv2.addWeighted(segmented_image, 0.7, overlay, 0.3, 0)
            
            # Draw contour
            cv2.drawContours(segmented_image, [largest_contour], -1, (0, 255, 0), 3)
            cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            return {
                "roof_mask": roof_mask,
                "roof_contour": largest_contour,
                "roof_area_pixels": float(roof_area_pixels),
                "roof_bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "segmented_image": segmented_image,
                "contours_found": len(roof_contours),
                "method": method,
                "success": True
            }
        else:
            return {
                "roof_mask": np.zeros(gray.shape, dtype=np.uint8),
                "roof_contour": None,
                "roof_area_pixels": 0.0,
                "roof_bbox": None,
                "segmented_image": img_array,
                "contours_found": 0,
                "method": method,
                "success": False
            }
            
    except Exception as e:
        print(f"Roof segmentation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "roof_mask": None,
            "roof_contour": None,
            "roof_area_pixels": 0.0,
            "roof_bbox": None,
            "segmented_image": np.array(image),
            "contours_found": 0,
            "method": method,
            "success": False,
            "error": str(e)
        }


def generate_roof_mask(image: Image.Image, segmentation_result: Dict) -> Dict:
    """
    Generate refined roof mask from segmentation results
    
    Args:
        image: PIL Image object
        segmentation_result: Result from segment_roof()
    
    Returns:
        Dictionary with roof mask and polygon data
    """
    try:
        roof_mask = segmentation_result.get("roof_mask")
        roof_contour = segmentation_result.get("roof_contour")
        
        if roof_mask is None or roof_contour is None:
            return {
                "roof_mask_base64": None,
                "roof_polygon": None,
                "mask_area_pixels": 0,
                "success": False
            }
        
        # Convert mask to PIL Image
        mask_image = Image.fromarray(roof_mask, mode='L')
        
        # Convert to base64
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_str = base64.b64encode(buffered.getvalue()).decode()
        mask_base64 = f"data:image/png;base64,{mask_str}"
        
        # Extract polygon from contour
        if len(roof_contour) > 0:
            # Simplify contour (reduce points)
            epsilon = 0.02 * cv2.arcLength(roof_contour, True)
            simplified_contour = cv2.approxPolyDP(roof_contour, epsilon, True)
            
            # Convert to polygon coordinates
            polygon = []
            for point in simplified_contour:
                polygon.append({"x": int(point[0][0]), "y": int(point[0][1])})
        else:
            polygon = None
        
        # Calculate mask area
        mask_area = int(np.sum(roof_mask == 255))
        
        return {
            "roof_mask_base64": mask_base64,
            "roof_polygon": polygon,
            "mask_area_pixels": mask_area,
            "contour_points": len(roof_contour) if roof_contour is not None else 0,
            "simplified_points": len(polygon) if polygon else 0,
            "success": True
        }
        
    except Exception as e:
        print(f"Roof mask generation error: {str(e)}")
        return {
            "roof_mask_base64": None,
            "roof_polygon": None,
            "mask_area_pixels": 0,
            "success": False,
            "error": str(e)
        }


def detect_roof_edges(image: Image.Image) -> Dict:
    """
    Detect roof edges using computer vision techniques
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with roof edge detection results
    """
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (remove small noise)
        min_area = (img_array.shape[0] * img_array.shape[1]) * 0.01  # 1% of image area
        roof_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Find largest contour (likely the roof)
        if roof_contours:
            largest_contour = max(roof_contours, key=cv2.contourArea)
            roof_area_pixels = cv2.contourArea(largest_contour)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw edges on image
            edge_image = img_array.copy()
            cv2.drawContours(edge_image, [largest_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(edge_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            return {
                "roof_contour_area_pixels": float(roof_area_pixels),
                "roof_bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "edge_image": edge_image,
                "contours_found": len(roof_contours)
            }
        else:
            return {
                "roof_contour_area_pixels": 0,
                "roof_bbox": None,
                "edge_image": img_array,
                "contours_found": 0
            }
    except Exception as e:
        print(f"Roof edge detection error: {str(e)}")
        return {
            "roof_contour_area_pixels": 0,
            "roof_bbox": None,
            "edge_image": np.array(image),
            "contours_found": 0,
            "error": str(e)
        }
