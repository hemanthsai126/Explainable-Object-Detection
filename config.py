"""
Configuration settings for Explainable Object Detection Framework
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class YOLOConfig:
    """YOLOv8 configuration"""
    model_size: str = "yolov8n.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "cuda"  # 'cuda' or 'cpu'
    

@dataclass
class CLIPConfig:
    """CLIP configuration"""
    model_name: str = "ViT-B/32"  # Options: ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
    device: str = "cuda"
    batch_size: int = 32


@dataclass
class LLaVAConfig:
    """LLaVA configuration"""
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    load_in_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    

@dataclass
class GradCAMConfig:
    """Grad-CAM configuration"""
    target_layers: List[str] = field(default_factory=lambda: ["model.model.model.9"])  # YOLOv8 backbone
    use_cuda: bool = True
    

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    occlusion_patch_size: int = 32
    occlusion_stride: int = 16
    num_samples_for_eval: int = 100
    save_visualizations: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    coco_root: str = "./data/coco"
    coco_annotations: str = "./data/coco/annotations"
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    

@dataclass
class Config:
    """Main configuration container"""
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    llava: LLaVAConfig = field(default_factory=LLaVAConfig)
    gradcam: GradCAMConfig = field(default_factory=GradCAMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        # Create necessary directories
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
        

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# CLIP text prompts for semantic similarity
def get_clip_prompts(class_name: str) -> List[str]:
    """Generate multiple text prompts for CLIP semantic matching"""
    return [
        f"a photo of a {class_name}",
        f"a {class_name} in the image",
        f"this is a {class_name}",
        f"an image showing a {class_name}",
        f"a clear view of a {class_name}",
    ]


# LLaVA prompt templates
LLAVA_SYSTEM_PROMPT = """You are an expert visual analyst providing concise, accurate explanations 
for object detection decisions. Focus on describing the specific visual features that identify the object."""

LLAVA_EXPLANATION_TEMPLATE = """Analyze this cropped region from an object detection system.
The detection model classified this as: {class_name} (confidence: {confidence:.2%})

Provide a brief explanation (2-3 sentences) describing:
1. The key visual features that support this classification
2. Any distinctive characteristics visible in the image

Be specific and focus only on what you can actually see in this image region."""

LLAVA_VERIFICATION_TEMPLATE = """Looking at this image region, the object detector identified it as a "{class_name}".
In 1-2 sentences, explain what visual evidence supports or contradicts this classification."""


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def update_config_for_cpu(config: Config) -> Config:
    """Update configuration for CPU-only inference"""
    config.yolo.device = "cpu"
    config.clip.device = "cpu"
    config.llava.device = "cpu"
    config.gradcam.use_cuda = False
    return config

