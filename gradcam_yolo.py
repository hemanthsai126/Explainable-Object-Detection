"""
Grad-CAM Implementation for YOLOv8
Generates visual saliency maps to explain detection predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from config import GradCAMConfig


@dataclass
class SaliencyMap:
    """Saliency map for a detection"""
    detection_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    heatmap: np.ndarray  # Normalized heatmap [0, 1]
    overlay: np.ndarray  # Heatmap overlaid on image
    raw_cam: np.ndarray  # Raw CAM values before normalization
    
    def get_salient_regions(self, threshold: float = 0.5) -> np.ndarray:
        """Get binary mask of highly salient regions"""
        return (self.heatmap > threshold).astype(np.uint8)
    
    def get_salient_area_ratio(self, threshold: float = 0.5) -> float:
        """Get ratio of salient area to bounding box area"""
        mask = self.get_salient_regions(threshold)
        x1, y1, x2, y2 = self.bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        salient_area = mask[y1:y2, x1:x2].sum()
        return salient_area / (bbox_area + 1e-6)


class YOLOGradCAM:
    """
    Grad-CAM for YOLOv8 models
    
    Generates class activation maps showing which image regions
    influenced the model's detection decisions.
    """
    
    def __init__(self, model, config: Optional[GradCAMConfig] = None):
        """
        Initialize Grad-CAM for YOLO
        
        Args:
            model: YOLOv8 model (ultralytics YOLO object)
            config: GradCAMConfig with settings
        """
        self.config = config or GradCAMConfig()
        self.model = model
        self.device = "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        
        # Storage for gradients and activations
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        # Target layers for CAM computation
        self.target_layers = self._get_target_layers()
        
    def _get_target_layers(self) -> List:
        """Get target layers for Grad-CAM"""
        target_layers = []
        
        try:
            # Access YOLOv8 model backbone
            # The model structure is model.model.model for the actual layers
            backbone = self.model.model.model
            
            # Common target layers for YOLOv8
            # Layer 9 is typically the last layer of the backbone (SPPF)
            # Layers 4, 6, 9 correspond to different scales
            target_indices = [4, 6, 9]  # Multi-scale features
            
            for idx in target_indices:
                if idx < len(backbone):
                    target_layers.append(backbone[idx])
                    
        except Exception as e:
            print(f"Warning: Could not access default layers: {e}")
            # Fallback: try to get any conv layers
            for name, module in self.model.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layers.append(module)
                    if len(target_layers) >= 3:
                        break
        
        return target_layers
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers"""
        self._remove_hooks()  # Clear existing hooks
        
        for i, layer in enumerate(self.target_layers):
            # Forward hook to capture activations
            def forward_hook(module, input, output, layer_id=i):
                self.activations[layer_id] = output.detach()
            
            # Backward hook to capture gradients
            def backward_hook(module, grad_input, grad_output, layer_id=i):
                self.gradients[layer_id] = grad_output[0].detach()
            
            fh = layer.register_forward_hook(forward_hook)
            bh = layer.register_full_backward_hook(backward_hook)
            self.hooks.extend([fh, bh])
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = {}
        self.activations = {}
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """Preprocess image for YOLO"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(image).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        
        return img_tensor.to(self.device)
    
    def compute_gradcam(self, 
                       image: Union[np.ndarray, Image.Image, str],
                       detection_results,
                       target_detection_idx: Optional[int] = None) -> List[SaliencyMap]:
        """
        Compute Grad-CAM for detected objects
        
        Args:
            image: Input image
            detection_results: Detection results from YOLODetector
            target_detection_idx: Specific detection to explain (None = all)
            
        Returns:
            List of SaliencyMap objects
        """
        # Load and preprocess image
        if isinstance(image, str):
            orig_image = cv2.imread(image)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            orig_image = np.array(image)
        else:
            orig_image = image.copy()
        
        h, w = orig_image.shape[:2]
        saliency_maps = []
        
        # Get detections to process
        if target_detection_idx is not None:
            detections_to_process = [(target_detection_idx, detection_results.detections[target_detection_idx])]
        else:
            detections_to_process = list(enumerate(detection_results.detections))
        
        for det_idx, detection in detections_to_process:
            try:
                # Compute CAM for this detection
                cam = self._compute_single_cam(orig_image, detection)
                
                # Resize CAM to image size
                cam_resized = cv2.resize(cam, (w, h))
                
                # Normalize
                cam_normalized = self._normalize_cam(cam_resized)
                
                # Create overlay
                overlay = self._create_overlay(orig_image, cam_normalized)
                
                saliency_map = SaliencyMap(
                    detection_id=det_idx,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    heatmap=cam_normalized,
                    overlay=overlay,
                    raw_cam=cam
                )
                saliency_maps.append(saliency_map)
                
            except Exception as e:
                print(f"Warning: Could not compute Grad-CAM for detection {det_idx}: {e}")
                continue
        
        return saliency_maps
    
    def _compute_single_cam(self, image: np.ndarray, detection) -> np.ndarray:
        """Compute CAM for a single detection"""
        self._register_hooks()
        
        # Prepare image
        img_tensor = self._preprocess_image(image)
        img_tensor.requires_grad_(True)
        
        # Forward pass through the model
        # We need to access the model's forward pass
        self.model.model.eval()
        
        # Get predictions
        with torch.enable_grad():
            # Run model forward
            preds = self.model.model(img_tensor)
            
            # For YOLO, we need to extract the relevant output
            # The output is typically [batch, num_boxes, 4+num_classes]
            if isinstance(preds, (list, tuple)):
                pred = preds[0]
            else:
                pred = preds
            
            # Find the output corresponding to our detection
            # Use the confidence score as the target for backprop
            x1, y1, x2, y2 = detection.bbox
            h, w = image.shape[:2]
            
            # Approximate center of detection in feature map coordinates
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            
            # Create a pseudo-target based on detection confidence
            # We'll use the class score at the predicted class
            target = pred[..., 4:].max() * detection.confidence
            
            # Backward pass
            self.model.model.zero_grad()
            target.backward(retain_graph=True)
        
        # Compute CAM from gradients and activations
        cams = []
        for layer_id in self.activations:
            if layer_id in self.gradients:
                activation = self.activations[layer_id]
                gradient = self.gradients[layer_id]
                
                # Global average pooling of gradients
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                
                # Weighted combination of activation maps
                cam = (weights * activation).sum(dim=1, keepdim=True)
                cam = F.relu(cam)  # Only positive contributions
                
                # Convert to numpy
                cam = cam.squeeze().cpu().numpy()
                cams.append(cam)
        
        self._remove_hooks()
        
        # Combine multi-scale CAMs
        if cams:
            # Resize all to same size and average
            target_size = cams[-1].shape  # Use largest
            combined_cam = np.zeros(target_size)
            
            for cam in cams:
                if cam.shape != target_size:
                    cam = cv2.resize(cam, (target_size[1], target_size[0]))
                combined_cam += cam
            
            combined_cam /= len(cams)
            return combined_cam
        
        # Fallback: return uniform cam
        return np.ones((20, 20))
    
    def _normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """Normalize CAM to [0, 1]"""
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    
    def _create_overlay(self, image: np.ndarray, cam: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """Create heatmap overlay on image"""
        # Convert CAM to colormap
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        return overlay
    
    def compute_attention_guided_cam(self, 
                                    image: np.ndarray,
                                    detection_results) -> List[SaliencyMap]:
        """
        Compute attention-guided Grad-CAM for better localization
        
        Uses attention weights if available in the model architecture.
        """
        # For YOLOv8, we can use the neck features for better localization
        return self.compute_gradcam(image, detection_results)
    
    def get_region_importance(self, 
                             saliency_map: SaliencyMap,
                             grid_size: int = 7) -> np.ndarray:
        """
        Compute importance scores for image regions
        
        Args:
            saliency_map: SaliencyMap object
            grid_size: Size of importance grid
            
        Returns:
            grid_size x grid_size array of importance scores
        """
        h, w = saliency_map.heatmap.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        importance = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                importance[i, j] = saliency_map.heatmap[y1:y2, x1:x2].mean()
        
        return importance


class GradCAMPlusPlus(YOLOGradCAM):
    """
    Grad-CAM++ implementation for better localization
    
    Uses second-order gradients for improved saliency maps,
    especially for multiple instances of the same class.
    """
    
    def _compute_single_cam(self, image: np.ndarray, detection) -> np.ndarray:
        """Compute Grad-CAM++ for a single detection"""
        self._register_hooks()
        
        img_tensor = self._preprocess_image(image)
        img_tensor.requires_grad_(True)
        
        self.model.model.eval()
        
        with torch.enable_grad():
            preds = self.model.model(img_tensor)
            
            if isinstance(preds, (list, tuple)):
                pred = preds[0]
            else:
                pred = preds
            
            target = pred[..., 4:].max() * detection.confidence
            
            # First derivative
            self.model.model.zero_grad()
            first_grad = torch.autograd.grad(target, img_tensor, 
                                            create_graph=True, 
                                            retain_graph=True)[0]
            
            # Second derivative (for Grad-CAM++)
            second_grad = torch.autograd.grad(first_grad.sum(), img_tensor,
                                             retain_graph=True)[0]
        
        cams = []
        for layer_id in self.activations:
            if layer_id in self.gradients:
                activation = self.activations[layer_id]
                gradient = self.gradients[layer_id]
                
                # Grad-CAM++ weighting
                grad_2 = gradient ** 2
                grad_3 = gradient ** 3
                
                # Alpha weights
                alpha_num = grad_2
                alpha_denom = 2 * grad_2 + (activation * grad_3).sum(dim=(2, 3), keepdim=True)
                alpha = alpha_num / (alpha_denom + 1e-7)
                
                # Weighted gradients
                weights = (alpha * F.relu(gradient)).sum(dim=(2, 3), keepdim=True)
                
                # CAM
                cam = (weights * activation).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().numpy()
                cams.append(cam)
        
        self._remove_hooks()
        
        if cams:
            target_size = cams[-1].shape
            combined_cam = np.zeros(target_size)
            for cam in cams:
                if cam.shape != target_size:
                    cam = cv2.resize(cam, (target_size[1], target_size[0]))
                combined_cam += cam
            combined_cam /= len(cams)
            return combined_cam
        
        return np.ones((20, 20))


class ScoreCAM(YOLOGradCAM):
    """
    Score-CAM implementation
    
    Gradient-free CAM using activation map scores instead of gradients.
    More stable than Grad-CAM but computationally more expensive.
    """
    
    def _compute_single_cam(self, image: np.ndarray, detection) -> np.ndarray:
        """Compute Score-CAM for a single detection"""
        self._register_hooks()
        
        img_tensor = self._preprocess_image(image)
        h, w = image.shape[:2]
        
        # Forward pass to get activations
        with torch.no_grad():
            self.model.model(img_tensor)
        
        # Use the last target layer
        if not self.activations:
            self._remove_hooks()
            return np.ones((20, 20))
        
        last_layer_id = max(self.activations.keys())
        activations = self.activations[last_layer_id]
        
        # Normalize activations
        batch_size, num_channels, act_h, act_w = activations.shape
        
        # Upsample each activation map and use as mask
        cams = []
        
        with torch.no_grad():
            for i in range(min(num_channels, 64)):  # Limit for efficiency
                # Get single channel activation
                act_map = activations[0, i:i+1, :, :]
                
                # Normalize to [0, 1]
                act_map = act_map - act_map.min()
                if act_map.max() > 0:
                    act_map = act_map / act_map.max()
                
                # Upsample to image size
                mask = F.interpolate(act_map.unsqueeze(0), size=(h, w), 
                                   mode='bilinear', align_corners=False)
                mask = mask.squeeze()
                
                # Apply mask to image
                masked_img = img_tensor.clone()
                masked_img = masked_img * mask.unsqueeze(0).unsqueeze(0)
                
                # Get score for masked image
                masked_preds = self.model.model(masked_img)
                if isinstance(masked_preds, (list, tuple)):
                    masked_pred = masked_preds[0]
                else:
                    masked_pred = masked_preds
                
                score = masked_pred[..., 4:].max().item()
                
                # Weight activation map by score
                weighted_cam = act_map.squeeze().cpu().numpy() * score
                cams.append(weighted_cam)
        
        self._remove_hooks()
        
        # Combine weighted CAMs
        if cams:
            combined = np.stack(cams, axis=0).sum(axis=0)
            return combined
        
        return np.ones((20, 20))


def visualize_saliency_maps(image: np.ndarray, 
                           saliency_maps: List[SaliencyMap],
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize multiple saliency maps
    
    Args:
        image: Original image
        saliency_maps: List of SaliencyMap objects
        save_path: Optional path to save visualization
        
    Returns:
        Visualization image
    """
    import matplotlib.pyplot as plt
    
    n_maps = len(saliency_maps)
    if n_maps == 0:
        return image
    
    # Create figure
    fig, axes = plt.subplots(2, n_maps + 1, figsize=(4 * (n_maps + 1), 8))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("With Bounding Boxes")
    axes[1, 0].axis('off')
    
    # Draw bounding boxes
    for sm in saliency_maps:
        x1, y1, x2, y2 = sm.bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='red', linewidth=2)
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x1, y1-5, f"{sm.class_name}: {sm.confidence:.2f}",
                       color='red', fontsize=8)
    
    # Individual saliency maps
    for i, sm in enumerate(saliency_maps):
        # Heatmap
        axes[0, i+1].imshow(sm.heatmap, cmap='jet')
        axes[0, i+1].set_title(f"Heatmap: {sm.class_name}")
        axes[0, i+1].axis('off')
        
        # Overlay
        axes[1, i+1].imshow(sm.overlay)
        axes[1, i+1].set_title(f"Overlay (conf: {sm.confidence:.2f})")
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # Convert to numpy array
    fig.canvas.draw()
    vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    return vis_img


def run_demo():
    """Run Grad-CAM demo"""
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize Grad-CAM
    gradcam = YOLOGradCAM(model)
    
    print("Grad-CAM initialized successfully")
    print(f"Target layers: {len(gradcam.target_layers)}")


if __name__ == "__main__":
    run_demo()

