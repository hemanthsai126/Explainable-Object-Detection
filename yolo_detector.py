"""
YOLOv8 Object Detection Module
Implements detection pipeline on MS COCO with region extraction
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from ultralytics import YOLO
import cv2

from config import YOLOConfig, COCO_CLASSES


@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    region: np.ndarray  # Cropped image region
    mask: Optional[np.ndarray] = None  # Segmentation mask if available
    
    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


@dataclass  
class DetectionResult:
    """Complete detection result for an image"""
    image: np.ndarray
    image_path: Optional[str]
    detections: List[Detection]
    raw_output: any  # Original YOLO output
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Filter detections by class name"""
        return [d for d in self.detections if d.class_name == class_name]
    
    def get_top_k_detections(self, k: int) -> List[Detection]:
        """Get top-k detections by confidence"""
        sorted_dets = sorted(self.detections, key=lambda x: x.confidence, reverse=True)
        return sorted_dets[:k]


class YOLODetector:
    """YOLOv8 Object Detection Pipeline"""
    
    def __init__(self, config: Optional[YOLOConfig] = None):
        """
        Initialize YOLOv8 detector
        
        Args:
            config: YOLOConfig object with model settings
        """
        self.config = config or YOLOConfig()
        self.device = self._setup_device()
        self.model = self._load_model()
        self.class_names = COCO_CLASSES
        
    def _setup_device(self) -> str:
        """Setup computation device"""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model"""
        print(f"Loading YOLOv8 model: {self.config.model_size}")
        model = YOLO(self.config.model_size)
        return model
    
    def _extract_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                       padding: float = 0.1) -> np.ndarray:
        """
        Extract region from image with optional padding
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding ratio to add around bbox
            
        Returns:
            Cropped image region
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        box_w, box_h = x2 - x1, y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return image[y1:y2, x1:x2].copy()
    
    def detect(self, image: Union[str, Path, np.ndarray, Image.Image],
              extract_regions: bool = True) -> DetectionResult:
        """
        Run object detection on an image
        
        Args:
            image: Image path, numpy array, or PIL Image
            extract_regions: Whether to extract cropped regions for each detection
            
        Returns:
            DetectionResult object containing all detections
        """
        # Load image
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
            
        # Run inference
        results = self.model(
            img_array,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        result = results[0]  # Single image
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
                # Extract region if requested
                region = None
                if extract_regions:
                    region = self._extract_region(img_array, (x1, y1, x2, y2))
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=float(conf),
                    region=region
                )
                detections.append(detection)
        
        return DetectionResult(
            image=img_array,
            image_path=image_path,
            detections=detections,
            raw_output=result
        )
    
    def detect_batch(self, images: List[Union[str, Path, np.ndarray]], 
                    extract_regions: bool = True) -> List[DetectionResult]:
        """
        Run detection on a batch of images
        
        Args:
            images: List of image paths or arrays
            extract_regions: Whether to extract cropped regions
            
        Returns:
            List of DetectionResult objects
        """
        return [self.detect(img, extract_regions) for img in images]
    
    def get_feature_maps(self, image: Union[str, Path, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for Grad-CAM
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of layer names to feature maps
        """
        # Load image
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
            
        feature_maps = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        # Register hooks on backbone layers
        try:
            # Access YOLOv8 backbone layers
            backbone = self.model.model.model
            for i, layer in enumerate(backbone):
                hook = layer.register_forward_hook(get_activation(f"layer_{i}"))
                hooks.append(hook)
        except Exception as e:
            print(f"Warning: Could not register hooks: {e}")
        
        # Run inference to trigger hooks
        with torch.no_grad():
            self.model(img_array, verbose=False)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return feature_maps
    
    def visualize_detections(self, result: DetectionResult, 
                            show_labels: bool = True,
                            show_confidence: bool = True,
                            line_thickness: int = 2) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            result: DetectionResult object
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            line_thickness: Box line thickness
            
        Returns:
            Annotated image
        """
        img = result.image.copy()
        
        # Color palette
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = tuple(map(int, colors[det.class_id]))
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw label
            if show_labels:
                label = det.class_name
                if show_confidence:
                    label = f"{label}: {det.confidence:.2f}"
                
                # Label background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img


class COCODataLoader:
    """Data loader for MS COCO dataset"""
    
    def __init__(self, coco_root: str, split: str = "val2017"):
        """
        Initialize COCO data loader
        
        Args:
            coco_root: Root directory of COCO dataset
            split: Dataset split (train2017, val2017)
        """
        self.coco_root = Path(coco_root)
        self.split = split
        self.images_dir = self.coco_root / split
        self.annotations_file = self.coco_root / "annotations" / f"instances_{split}.json"
        
        self.coco = None
        self.image_ids = []
        
        self._load_annotations()
    
    def _load_annotations(self):
        """Load COCO annotations"""
        try:
            from pycocotools.coco import COCO
            if self.annotations_file.exists():
                self.coco = COCO(str(self.annotations_file))
                self.image_ids = list(self.coco.imgs.keys())
                print(f"Loaded {len(self.image_ids)} images from COCO {self.split}")
            else:
                print(f"Annotations file not found: {self.annotations_file}")
                self._load_images_only()
        except ImportError:
            print("pycocotools not installed, loading images only")
            self._load_images_only()
    
    def _load_images_only(self):
        """Load images without annotations"""
        if self.images_dir.exists():
            self.image_ids = [f.stem for f in self.images_dir.glob("*.jpg")]
            print(f"Found {len(self.image_ids)} images in {self.images_dir}")
    
    def get_image_path(self, image_id: Union[int, str]) -> Path:
        """Get full path to image"""
        if isinstance(image_id, int):
            image_id = f"{image_id:012d}"
        return self.images_dir / f"{image_id}.jpg"
    
    def get_image(self, image_id: Union[int, str]) -> np.ndarray:
        """Load image by ID"""
        path = self.get_image_path(image_id)
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_ground_truth(self, image_id: int) -> List[Dict]:
        """Get ground truth annotations for image"""
        if self.coco is None:
            return []
        
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append({
                'bbox': (int(x), int(y), int(x+w), int(y+h)),
                'category_id': ann['category_id'],
                'category_name': self.coco.cats[ann['category_id']]['name'],
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0)
            })
        
        return gt_boxes
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __iter__(self):
        for img_id in self.image_ids:
            yield self.get_image_path(img_id)


def run_demo():
    """Run a demo of the YOLO detector"""
    import matplotlib.pyplot as plt
    
    # Initialize detector
    config = YOLOConfig(model_size="yolov8n.pt", confidence_threshold=0.3)
    detector = YOLODetector(config)
    
    # Create a test image (colored rectangle pattern)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[100:300, 200:400] = [255, 0, 0]  # Red rectangle
    
    # Run detection
    result = detector.detect(test_img)
    
    print(f"Found {result.num_detections} detections")
    for det in result.detections:
        print(f"  - {det}")
    
    # Visualize
    vis_img = detector.visualize_detections(result)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.title(f"YOLO Detection Results ({result.num_detections} objects)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("demo_detection.png")
    print("Saved visualization to demo_detection.png")


if __name__ == "__main__":
    run_demo()

