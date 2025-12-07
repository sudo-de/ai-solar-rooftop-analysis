import torch
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import logging

logger = logging.getLogger(__name__)

class RoofSegmentationService:
    """AI service for detecting exact roof outlines using SegFormer"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load SegFormer model for semantic segmentation
        # Try B1 model first for better accuracy, fallback to B0
        try:
            # Try B1 with correct model name
            model_name = "nvidia/segformer-b1-finetuned-ade-512-512"
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            logger.info("Loaded SegFormer-B1 model (higher accuracy)")
        except Exception as e:
            logger.warning(f"Could not load SegFormer-B1, falling back to B0: {e}")
            model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()

        # ADE20K class labels (SegFormer was trained on this dataset)
        self.labels = [
            "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
            "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
            "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting",
            "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
            "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing",
            "cushion", "base", "box", "column", "signboard", "chest of drawers",
            "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator",
            "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow",
            "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table",
            "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm",
            "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
            "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning",
            "streetlight", "booth", "television receiver", "airplane", "dirt track",
            "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle",
            "buffet", "poster", "stage", "van", "ship", "fountain", "conveyor belt",
            "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket",
            "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
            "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle",
            "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
            "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
            "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock",
            "flag"
        ]

    def segment_roof(self, image_path: str) -> dict:
        """Detect and segment roof areas from the image"""
        try:
            # Load and preprocess image with enhancements
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # Apply image preprocessing for better results
            image = self._preprocess_image(image)

            # Process image for SegFormer
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Upsample logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=original_size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False
            )

            # Get segmentation map
            predicted_segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # Find roof/building areas (class indices for buildings/houses)
            # Expanded class list for better detection
            roof_classes = [1, 25, 46, 0]  # building, house, skyscraper, wall
            roof_mask = np.isin(predicted_segmentation, roof_classes)

            # Post-process mask with morphological operations for better quality
            roof_mask_uint8 = (roof_mask * 255).astype(np.uint8)
            roof_mask_uint8 = self._postprocess_mask(roof_mask_uint8)

            # Find contours of roof areas
            contours, _ = cv2.findContours(roof_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate roof metrics
            roof_area_pixels = np.sum(roof_mask)
            image_area_pixels = original_size[0] * original_size[1]
            roof_percentage = (roof_area_pixels / image_area_pixels) * 100

            # Get the largest roof contour (assuming it's the main building)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                roof_perimeter = cv2.arcLength(largest_contour, True)
                roof_area_contour = cv2.contourArea(largest_contour)

                # Create detailed roof outline
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Convert contour to coordinates
                roof_outline = approx_contour.reshape(-1, 2).tolist()
            else:
                roof_perimeter = 0
                roof_area_contour = 0
                roof_outline = []

            # Create segmented image overlay
            segmented_image = self._create_segmented_overlay(image, roof_mask)

            # Convert to base64
            buffered = BytesIO()
            segmented_image.save(buffered, format="PNG")
            segmented_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Determine model version
            model_version = "SegFormer-B1" if hasattr(self.model.config, 'model_type') and 'b1' in str(self.model.config.model_type).lower() else "SegFormer-B0"
            
            return {
                "roof_detected": len(contours) > 0,
                "roof_area_pixels": int(roof_area_pixels),
                "roof_percentage": round(roof_percentage, 2),
                "roof_perimeter_pixels": round(roof_perimeter, 2),
                "roof_outline_coordinates": roof_outline,
                "segmented_image_base64": f"data:image/png;base64,{segmented_base64}",
                "confidence_score": round(float(torch.softmax(logits, dim=1).max().item()), 3),
                "model_used": model_version,
                "processing_time_seconds": 0.0  # Will be calculated by caller
            }

        except Exception as e:
            logger.error(f"Error in roof segmentation: {str(e)}")
            model_version = "SegFormer-B1" if hasattr(self, 'model') and hasattr(self.model.config, 'model_type') and 'b1' in str(self.model.config.model_type).lower() else "SegFormer-B0"
            
            return {
                "roof_detected": False,
                "error": str(e),
                "roof_area_pixels": 0,
                "roof_percentage": 0.0,
                "roof_perimeter_pixels": 0.0,
                "roof_outline_coordinates": [],
                "segmented_image_base64": "",
                "confidence_score": 0.0,
                "model_used": model_version,
                "processing_time_seconds": 0.0
            }

    def _create_segmented_overlay(self, original_image: Image.Image, roof_mask: np.ndarray) -> Image.Image:
        """Create a visual overlay showing the detected roof areas"""
        # Convert PIL to numpy array
        img_array = np.array(original_image)

        # Create colored overlay
        overlay = img_array.copy()
        overlay[roof_mask] = overlay[roof_mask] * 0.7 + np.array([255, 0, 0]) * 0.3  # Red tint for roof

        # Add contour lines
        contours, _ = cv2.findContours((roof_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)  # Green contours

        return Image.fromarray(overlay)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better segmentation results"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            if len(img_array.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                # Merge channels and convert back to RGB
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                img_array = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # Normalize image
            img_array = img_array.astype(np.float32) / 255.0
            img_array = (img_array * 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
        except Exception as e:
            logger.warning(f"Error in image preprocessing: {e}")
            return image

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process segmentation mask for better quality"""
        try:
            # Remove small noise with morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill small holes with morphological closing
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Remove small connected components (noise)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Keep only components larger than 1% of image area
            min_area = (mask.shape[0] * mask.shape[1]) * 0.01
            cleaned_mask = np.zeros_like(mask)
            
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned_mask[labels == i] = 255
            
            return cleaned_mask
        except Exception as e:
            logger.warning(f"Error in mask post-processing: {e}")
            return mask
