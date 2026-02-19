"""
Explainable Object Detection Pipeline
Integrates YOLOv8, CLIP, LLaVA, and Grad-CAM for interpretable detection
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import warnings

from config import (
    Config, 
    get_default_config, 
    update_config_for_cpu,
    COCO_CLASSES
)
from yolo_detector import YOLODetector, DetectionResult, Detection, COCODataLoader
from clip_embeddings import CLIPEmbedder, RegionEmbedding, SemanticMatch
from gradcam_yolo import YOLOGradCAM, SaliencyMap, GradCAMPlusPlus
from llava_explainer import LLaVAExplainer, Explanation, SimpleLLMExplainer
from evaluation import (
    ExplanationEvaluator, 
    EvaluationResult, 
    EvaluationReport,
    AlignmentMetrics,
    FaithfulnessMetrics,
    ExplanationQuality
)


@dataclass
class ExplainedDetection:
    """Complete explained detection with all analysis"""
    detection: Detection
    semantic_analysis: Optional[RegionEmbedding] = None
    saliency_map: Optional[SaliencyMap] = None
    explanation: Optional[Explanation] = None
    evaluation: Optional[EvaluationResult] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = {
            'bbox': self.detection.bbox,
            'class_name': self.detection.class_name,
            'confidence': self.detection.confidence,
        }
        
        if self.semantic_analysis and self.semantic_analysis.semantic_match:
            result['semantic'] = {
                'alignment_score': self.semantic_analysis.semantic_match.alignment_score,
                'top_clip_classes': self.semantic_analysis.semantic_match.top_k_classes[:3],
            }
        
        if self.explanation:
            result['explanation'] = {
                'text': self.explanation.explanation_text,
                'visual_features': self.explanation.visual_features,
                'uncertainty': self.explanation.uncertainty_indicators,
            }
        
        if self.evaluation:
            result['evaluation'] = self.evaluation.to_dict()
        
        return result


@dataclass
class PipelineResult:
    """Complete pipeline result for an image"""
    image_path: Optional[str]
    image: np.ndarray
    detections: List[ExplainedDetection]
    processing_time: float
    metadata: Dict = field(default_factory=dict)
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    def get_high_confidence_detections(self, threshold: float = 0.7) -> List[ExplainedDetection]:
        """Get detections above confidence threshold"""
        return [d for d in self.detections if d.detection.confidence >= threshold]
    
    def get_well_aligned_detections(self, threshold: float = 0.6) -> List[ExplainedDetection]:
        """Get detections with high CLIP alignment"""
        return [d for d in self.detections 
                if d.semantic_analysis and 
                d.semantic_analysis.semantic_match and
                d.semantic_analysis.semantic_match.alignment_score >= threshold]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'image_path': self.image_path,
            'num_detections': self.num_detections,
            'processing_time': self.processing_time,
            'detections': [d.to_dict() for d in self.detections],
            'metadata': self.metadata
        }
    
    def save(self, filepath: str):
        """Save results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ExplainableDetectionPipeline:
    """
    Main pipeline for explainable object detection
    
    Integrates:
    - YOLOv8 for object detection
    - CLIP for semantic grounding
    - Grad-CAM for visual saliency
    - LLaVA for natural language explanations
    """
    
    def __init__(self, config: Optional[Config] = None, use_llava: bool = True):
        """
        Initialize the explainable detection pipeline
        
        Args:
            config: Configuration object
            use_llava: Whether to load LLaVA model (requires significant memory)
        """
        self.config = config or get_default_config()
        self.use_llava = use_llava
        
        # Initialize components
        print("Initializing Explainable Detection Pipeline...")
        print("=" * 60)
        
        self._init_detector()
        self._init_clip()
        self._init_gradcam()
        
        if use_llava:
            self._init_llava()
        else:
            self._init_simple_explainer()
        
        self._init_evaluator()
        
        print("=" * 60)
        print("Pipeline initialization complete!")
    
    def _init_detector(self):
        """Initialize YOLO detector"""
        print("Loading YOLOv8 detector...")
        self.detector = YOLODetector(self.config.yolo)
    
    def _init_clip(self):
        """Initialize CLIP embedder"""
        print("Loading CLIP model...")
        self.clip = CLIPEmbedder(self.config.clip)
        self.clip.compute_class_embeddings()
    
    def _init_gradcam(self):
        """Initialize Grad-CAM"""
        print("Initializing Grad-CAM...")
        self.gradcam = YOLOGradCAM(self.detector.model, self.config.gradcam)
    
    def _init_llava(self):
        """Initialize LLaVA explainer"""
        print("Loading LLaVA model (this may take a while)...")
        try:
            self.explainer = LLaVAExplainer(self.config.llava)
        except Exception as e:
            print(f"Warning: Could not load LLaVA: {e}")
            print("Falling back to simple explainer")
            self._init_simple_explainer()
            self.use_llava = False
    
    def _init_simple_explainer(self):
        """Initialize simple rule-based explainer"""
        print("Using simple rule-based explainer")
        self.explainer = SimpleLLMExplainer()
    
    def _init_evaluator(self):
        """Initialize evaluator"""
        print("Initializing evaluator...")
        self.evaluator = ExplanationEvaluator(self.config.evaluation)
    
    def process_image(self, 
                     image: Union[str, Path, np.ndarray, Image.Image],
                     max_detections: int = 10,
                     generate_explanations: bool = True,
                     compute_saliency: bool = True,
                     run_evaluation: bool = True) -> PipelineResult:
        """
        Process a single image through the complete pipeline
        
        Args:
            image: Input image (path, array, or PIL Image)
            max_detections: Maximum number of detections to explain
            generate_explanations: Whether to generate text explanations
            compute_saliency: Whether to compute Grad-CAM saliency maps
            run_evaluation: Whether to run evaluation metrics
            
        Returns:
            PipelineResult with all analysis
        """
        import time
        start_time = time.time()
        
        # Load image
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
        
        # Step 1: Object Detection
        print("Step 1: Running object detection...")
        detection_result = self.detector.detect(image)
        
        if detection_result.num_detections == 0:
            print("No detections found")
            return PipelineResult(
                image_path=image_path,
                image=detection_result.image,
                detections=[],
                processing_time=time.time() - start_time
            )
        
        print(f"  Found {detection_result.num_detections} objects")
        
        # Limit detections for detailed analysis
        detections = detection_result.detections[:max_detections]
        
        explained_detections = []
        
        for i, det in enumerate(tqdm(detections, desc="Analyzing detections")):
            # Step 2: CLIP Semantic Analysis
            print(f"\nStep 2: CLIP analysis for {det.class_name}...")
            if det.region is not None:
                semantic_match = self.clip.compute_semantic_similarity(
                    det.region, det.class_name, det.confidence
                )
                semantic_analysis = RegionEmbedding(
                    region_id=i,
                    image_embedding=self.clip.encode_image(det.region).cpu(),
                    class_name=det.class_name,
                    confidence=det.confidence,
                    semantic_match=semantic_match
                )
            else:
                semantic_analysis = None
            
            # Step 3: Grad-CAM Saliency
            saliency_map = None
            if compute_saliency:
                print(f"Step 3: Computing Grad-CAM saliency...")
                try:
                    saliency_maps = self.gradcam.compute_gradcam(
                        detection_result.image, 
                        detection_result,
                        target_detection_idx=i
                    )
                    if saliency_maps:
                        saliency_map = saliency_maps[0]
                except Exception as e:
                    print(f"  Warning: Grad-CAM failed: {e}")
            
            # Step 4: Generate Explanation
            explanation = None
            if generate_explanations and det.region is not None:
                print(f"Step 4: Generating explanation...")
                try:
                    explanation = self.explainer.generate_explanation(
                        det.region, det.class_name, det.confidence
                    )
                    explanation.detection_id = i
                except Exception as e:
                    print(f"  Warning: Explanation generation failed: {e}")
            
            # Step 5: Evaluation
            evaluation = None
            if run_evaluation and saliency_map and explanation:
                print(f"Step 5: Evaluating explanation quality...")
                try:
                    evaluation = self.evaluator.evaluate_detection(
                        detection_result.image,
                        det,
                        saliency_map,
                        explanation,
                        self.detector
                    )
                except Exception as e:
                    print(f"  Warning: Evaluation failed: {e}")
            
            explained_detection = ExplainedDetection(
                detection=det,
                semantic_analysis=semantic_analysis,
                saliency_map=saliency_map,
                explanation=explanation,
                evaluation=evaluation
            )
            explained_detections.append(explained_detection)
        
        processing_time = time.time() - start_time
        
        return PipelineResult(
            image_path=image_path,
            image=detection_result.image,
            detections=explained_detections,
            processing_time=processing_time,
            metadata={
                'yolo_model': self.config.yolo.model_size,
                'clip_model': self.config.clip.model_name,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def process_batch(self,
                     images: List[Union[str, Path, np.ndarray]],
                     max_detections_per_image: int = 5,
                     **kwargs) -> List[PipelineResult]:
        """
        Process a batch of images
        
        Args:
            images: List of images to process
            max_detections_per_image: Max detections to explain per image
            **kwargs: Additional arguments for process_image
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        for img in tqdm(images, desc="Processing images"):
            try:
                result = self.process_image(
                    img, 
                    max_detections=max_detections_per_image,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        return results
    
    def analyze_misalignments(self, result: PipelineResult) -> List[Dict]:
        """
        Analyze cases where YOLO and CLIP disagree
        
        Args:
            result: PipelineResult to analyze
            
        Returns:
            List of mismatch analysis dictionaries
        """
        mismatches = []
        
        for det in result.detections:
            if det.semantic_analysis and det.semantic_analysis.semantic_match:
                match = det.semantic_analysis.semantic_match
                
                # Check if CLIP's top prediction differs from YOLO's
                if match.top_k_classes:
                    clip_top = match.top_k_classes[0][0]
                    
                    if clip_top != match.predicted_class:
                        mismatch = {
                            'bbox': det.detection.bbox,
                            'yolo_class': match.predicted_class,
                            'yolo_confidence': match.predicted_confidence,
                            'clip_top_class': clip_top,
                            'clip_score': match.top_k_classes[0][1],
                            'alignment_score': match.alignment_score,
                        }
                        
                        # Add LLaVA analysis if available
                        if self.use_llava and det.detection.region is not None:
                            try:
                                comparison = self.explainer.compare_explanations(
                                    det.detection.region,
                                    match.predicted_class,
                                    clip_top,
                                    match.predicted_confidence
                                )
                                mismatch['llava_analysis'] = comparison
                            except:
                                pass
                        
                        mismatches.append(mismatch)
        
        return mismatches
    
    def run_faithfulness_analysis(self, 
                                 image: Union[str, np.ndarray],
                                 result: PipelineResult) -> Dict:
        """
        Run comprehensive faithfulness analysis
        
        Performs occlusion tests to verify that saliency maps
        correctly identify important regions.
        
        Args:
            image: Original image
            result: PipelineResult with saliency maps
            
        Returns:
            Faithfulness analysis dictionary
        """
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        analysis = {
            'num_detections': result.num_detections,
            'detections': []
        }
        
        for det in result.detections:
            det_analysis = {
                'class': det.detection.class_name,
                'original_confidence': det.detection.confidence,
            }
            
            if det.saliency_map:
                # Test 1: Occlude salient regions
                salient_conf = self._test_occlusion(
                    image, det.saliency_map.heatmap, 
                    det.detection, occlude_salient=True
                )
                
                # Test 2: Occlude non-salient regions
                non_salient_conf = self._test_occlusion(
                    image, det.saliency_map.heatmap,
                    det.detection, occlude_salient=False
                )
                
                det_analysis.update({
                    'conf_after_salient_occlusion': salient_conf,
                    'conf_after_non_salient_occlusion': non_salient_conf,
                    'salient_drop': det.detection.confidence - salient_conf,
                    'non_salient_drop': det.detection.confidence - non_salient_conf,
                    'faithfulness_score': (det.detection.confidence - salient_conf) / 
                                         (det.detection.confidence + 1e-6)
                })
            
            analysis['detections'].append(det_analysis)
        
        # Summary
        if analysis['detections']:
            faith_scores = [d.get('faithfulness_score', 0) for d in analysis['detections']]
            analysis['mean_faithfulness'] = np.mean(faith_scores)
            analysis['std_faithfulness'] = np.std(faith_scores)
        
        return analysis
    
    def _test_occlusion(self, image: np.ndarray, heatmap: np.ndarray,
                       detection: Detection, occlude_salient: bool = True) -> float:
        """Test detection confidence after occlusion"""
        import cv2
        
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Create mask
        if occlude_salient:
            mask = heatmap_resized > 0.5
        else:
            mask = heatmap_resized < 0.3
        
        if not mask.any():
            return detection.confidence
        
        # Occlude
        occluded = image.copy()
        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
        occluded[mask] = mean_color
        
        # Re-detect
        try:
            result = self.detector.detect(occluded, extract_regions=False)
            
            # Find matching detection
            x1, y1, x2, y2 = detection.bbox
            
            for det in result.detections:
                if det.class_name == detection.class_name:
                    dx1, dy1, dx2, dy2 = det.bbox
                    
                    # Check overlap
                    xi1 = max(x1, dx1)
                    yi1 = max(y1, dy1)
                    xi2 = min(x2, dx2)
                    yi2 = min(y2, dy2)
                    
                    if xi2 > xi1 and yi2 > yi1:
                        inter = (xi2 - xi1) * (yi2 - yi1)
                        area1 = (x2 - x1) * (y2 - y1)
                        
                        if inter / area1 > 0.3:
                            return det.confidence
            
            return 0.0  # Detection lost
            
        except Exception:
            return detection.confidence
    
    def generate_report(self, results: List[PipelineResult], 
                       output_path: str) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: List of pipeline results
            output_path: Path to save report
            
        Returns:
            Report dictionary
        """
        all_evaluations = []
        
        for result in results:
            for det in result.detections:
                if det.evaluation:
                    all_evaluations.append(det.evaluation)
        
        if not all_evaluations:
            print("No evaluations to report")
            return {}
        
        report = EvaluationReport(all_evaluations)
        summary = report.generate_summary()
        
        # Save report
        report.save_report(output_path)
        
        return summary
    
    def visualize_result(self, result: PipelineResult, 
                        save_path: Optional[str] = None) -> np.ndarray:
        """
        Create comprehensive visualization of results
        
        Args:
            result: PipelineResult to visualize
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        n_dets = min(len(result.detections), 4)
        if n_dets == 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(result.image)
            ax.set_title("No detections found")
            ax.axis('off')
        else:
            fig = plt.figure(figsize=(16, 4 * n_dets))
            
            for i, det in enumerate(result.detections[:n_dets]):
                # Row for each detection: [image+bbox, saliency, region, explanation]
                
                # 1. Image with bounding box
                ax1 = fig.add_subplot(n_dets, 4, i*4 + 1)
                ax1.imshow(result.image)
                x1, y1, x2, y2 = det.detection.bbox
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='lime', linewidth=2)
                ax1.add_patch(rect)
                ax1.set_title(f"{det.detection.class_name}\n(conf: {det.detection.confidence:.2f})")
                ax1.axis('off')
                
                # 2. Saliency map
                ax2 = fig.add_subplot(n_dets, 4, i*4 + 2)
                if det.saliency_map:
                    ax2.imshow(det.saliency_map.overlay)
                    ax2.set_title("Grad-CAM Saliency")
                else:
                    ax2.imshow(result.image)
                    ax2.set_title("Saliency N/A")
                ax2.axis('off')
                
                # 3. Cropped region
                ax3 = fig.add_subplot(n_dets, 4, i*4 + 3)
                if det.detection.region is not None:
                    ax3.imshow(det.detection.region)
                    
                    # Add CLIP info
                    if det.semantic_analysis and det.semantic_analysis.semantic_match:
                        match = det.semantic_analysis.semantic_match
                        ax3.set_title(f"CLIP align: {match.alignment_score:.2f}\n"
                                     f"Top: {match.top_k_classes[0][0] if match.top_k_classes else 'N/A'}")
                    else:
                        ax3.set_title("Detected Region")
                else:
                    ax3.text(0.5, 0.5, "Region N/A", ha='center', va='center')
                ax3.axis('off')
                
                # 4. Explanation text
                ax4 = fig.add_subplot(n_dets, 4, i*4 + 4)
                ax4.axis('off')
                
                if det.explanation:
                    text = det.explanation.explanation_text
                    # Wrap text
                    wrapped = '\n'.join([text[j:j+40] for j in range(0, len(text), 40)])
                    ax4.text(0.05, 0.95, wrapped, transform=ax4.transAxes,
                            fontsize=9, verticalalignment='top', wrap=True,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    if det.evaluation:
                        score = det.evaluation.get_overall_score()
                        ax4.text(0.05, 0.05, f"Quality: {score:.2f}",
                                transform=ax4.transAxes, fontsize=10)
                else:
                    ax4.text(0.5, 0.5, "No explanation", ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Convert to numpy
        fig.canvas.draw()
        vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return vis_img


def create_pipeline(use_gpu: bool = True, use_llava: bool = False) -> ExplainableDetectionPipeline:
    """
    Factory function to create pipeline with appropriate settings
    
    Args:
        use_gpu: Whether to use GPU
        use_llava: Whether to load LLaVA model
        
    Returns:
        Configured ExplainableDetectionPipeline
    """
    config = get_default_config()
    
    if not use_gpu or not torch.cuda.is_available():
        config = update_config_for_cpu(config)
        print("Running on CPU")
    else:
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    
    return ExplainableDetectionPipeline(config, use_llava=use_llava)


def run_demo():
    """Run pipeline demo"""
    print("\n" + "=" * 60)
    print("EXPLAINABLE OBJECT DETECTION PIPELINE DEMO")
    print("=" * 60)
    
    # Create pipeline (without LLaVA for quick demo)
    print("\nInitializing pipeline...")
    pipeline = create_pipeline(use_gpu=True, use_llava=False)
    
    # Create a test image with some shapes
    print("\nCreating test image...")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored shapes
    import cv2
    cv2.rectangle(test_img, (100, 100), (250, 250), (255, 0, 0), -1)  # Red rect
    cv2.circle(test_img, (450, 200), 80, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(test_img, (300, 300), (500, 450), (0, 0, 255), -1)  # Blue rect
    
    # Process image
    print("\nProcessing image...")
    result = pipeline.process_image(
        test_img,
        max_detections=5,
        generate_explanations=True,
        compute_saliency=True,
        run_evaluation=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Number of detections: {result.num_detections}")
    
    for i, det in enumerate(result.detections):
        print(f"\n--- Detection {i+1} ---")
        print(f"Class: {det.detection.class_name}")
        print(f"Confidence: {det.detection.confidence:.2%}")
        print(f"BBox: {det.detection.bbox}")
        
        if det.semantic_analysis and det.semantic_analysis.semantic_match:
            print(f"CLIP Alignment: {det.semantic_analysis.semantic_match.alignment_score:.2f}")
        
        if det.explanation:
            print(f"Explanation: {det.explanation.explanation_text[:100]}...")
        
        if det.evaluation:
            print(f"Quality Score: {det.evaluation.get_overall_score():.2f}")
    
    # Save visualization
    print("\nSaving visualization...")
    os.makedirs("outputs", exist_ok=True)
    pipeline.visualize_result(result, "outputs/demo_result.png")
    
    # Save JSON results
    result.save("outputs/demo_result.json")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check outputs/ folder for results.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

