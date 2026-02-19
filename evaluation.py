"""
Evaluation Metrics Module
Measures alignment, faithfulness, and quality of explanations
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm

from config import EvaluationConfig


@dataclass
class AlignmentMetrics:
    """Metrics for explanation-saliency alignment"""
    text_saliency_overlap: float  # Overlap between mentioned features and salient regions
    spatial_consistency: float  # How well spatial references match saliency
    feature_coverage: float  # Proportion of salient features mentioned in explanation
    keyword_precision: float  # Precision of visual keywords in explanation
    keyword_recall: float  # Recall of visual keywords
    
    def to_dict(self) -> Dict:
        return {
            'text_saliency_overlap': self.text_saliency_overlap,
            'spatial_consistency': self.spatial_consistency,
            'feature_coverage': self.feature_coverage,
            'keyword_precision': self.keyword_precision,
            'keyword_recall': self.keyword_recall,
            'mean_alignment': np.mean([
                self.text_saliency_overlap,
                self.spatial_consistency,
                self.feature_coverage
            ])
        }


@dataclass
class FaithfulnessMetrics:
    """Metrics for explanation faithfulness"""
    occlusion_sensitivity: float  # Confidence drop when occluding salient regions
    counterfactual_consistency: float  # Consistency with counterfactual tests
    feature_attribution_agreement: float  # Agreement between explanation and Grad-CAM
    prediction_alignment: float  # How well explanation aligns with model prediction
    
    def to_dict(self) -> Dict:
        return {
            'occlusion_sensitivity': self.occlusion_sensitivity,
            'counterfactual_consistency': self.counterfactual_consistency,
            'feature_attribution_agreement': self.feature_attribution_agreement,
            'prediction_alignment': self.prediction_alignment,
            'mean_faithfulness': np.mean([
                self.occlusion_sensitivity,
                self.counterfactual_consistency,
                self.feature_attribution_agreement
            ])
        }


@dataclass
class ExplanationQuality:
    """Metrics for explanation text quality"""
    specificity: float  # How specific the explanation is
    conciseness: float  # Brevity without loss of information
    coherence: float  # Logical flow and structure
    grounding: float  # References to visual evidence
    
    def to_dict(self) -> Dict:
        return {
            'specificity': self.specificity,
            'conciseness': self.conciseness,
            'coherence': self.coherence,
            'grounding': self.grounding,
            'mean_quality': np.mean([
                self.specificity,
                self.conciseness,
                self.coherence,
                self.grounding
            ])
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a detection"""
    detection_id: int
    class_name: str
    alignment: AlignmentMetrics
    faithfulness: FaithfulnessMetrics
    quality: ExplanationQuality
    metadata: Dict = field(default_factory=dict)
    
    def get_overall_score(self) -> float:
        """Compute overall explanation quality score"""
        alignment_score = np.mean([
            self.alignment.text_saliency_overlap,
            self.alignment.spatial_consistency,
            self.alignment.feature_coverage
        ])
        
        faithfulness_score = np.mean([
            self.faithfulness.occlusion_sensitivity,
            self.faithfulness.feature_attribution_agreement,
            self.faithfulness.prediction_alignment
        ])
        
        quality_score = np.mean([
            self.quality.specificity,
            self.quality.coherence,
            self.quality.grounding
        ])
        
        # Weighted combination
        return 0.4 * alignment_score + 0.4 * faithfulness_score + 0.2 * quality_score
    
    def to_dict(self) -> Dict:
        return {
            'detection_id': self.detection_id,
            'class_name': self.class_name,
            'alignment': self.alignment.to_dict(),
            'faithfulness': self.faithfulness.to_dict(),
            'quality': self.quality.to_dict(),
            'overall_score': self.get_overall_score(),
            'metadata': self.metadata
        }


class ExplanationEvaluator:
    """
    Evaluator for explainable object detection
    
    Measures:
    1. Alignment between explanations and saliency maps
    2. Faithfulness via occlusion testing
    3. Explanation text quality
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator
        
        Args:
            config: EvaluationConfig with evaluation settings
        """
        self.config = config or EvaluationConfig()
        
        # Visual feature vocabulary for matching
        self.visual_vocabulary = self._build_visual_vocabulary()
        
        # Spatial reference patterns
        self.spatial_patterns = self._build_spatial_patterns()
    
    def _build_visual_vocabulary(self) -> Dict[str, List[str]]:
        """Build vocabulary of visual features by category"""
        return {
            'color': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
                     'brown', 'black', 'white', 'gray', 'grey', 'golden', 'silver'],
            'shape': ['round', 'circular', 'square', 'rectangular', 'oval', 'triangular',
                     'elongated', 'curved', 'straight', 'angular', 'pointed'],
            'texture': ['smooth', 'rough', 'furry', 'fuzzy', 'shiny', 'matte', 'glossy',
                       'textured', 'patterned', 'striped', 'spotted', 'dotted'],
            'size': ['large', 'small', 'medium', 'tiny', 'huge', 'big', 'compact',
                    'tall', 'short', 'wide', 'narrow', 'thick', 'thin'],
            'part': ['head', 'body', 'legs', 'arms', 'tail', 'ears', 'eyes', 'nose',
                    'mouth', 'wheel', 'window', 'door', 'roof', 'wing', 'handle']
        }
    
    def _build_spatial_patterns(self) -> List[str]:
        """Build patterns for spatial references"""
        return [
            'top', 'bottom', 'left', 'right', 'center', 'middle',
            'upper', 'lower', 'front', 'back', 'side',
            'corner', 'edge', 'border', 'background', 'foreground'
        ]
    
    def evaluate_alignment(self,
                          explanation,  # Explanation object
                          saliency_map,  # SaliencyMap object
                          image: np.ndarray) -> AlignmentMetrics:
        """
        Evaluate alignment between explanation text and saliency map
        
        Args:
            explanation: Explanation object with text
            saliency_map: SaliencyMap object with heatmap
            image: Original image
            
        Returns:
            AlignmentMetrics object
        """
        text = explanation.explanation_text.lower()
        heatmap = saliency_map.heatmap
        
        # 1. Text-saliency overlap
        # Check if mentioned features correspond to salient regions
        text_saliency = self._compute_text_saliency_overlap(text, heatmap, image)
        
        # 2. Spatial consistency
        spatial_score = self._evaluate_spatial_references(text, heatmap)
        
        # 3. Feature coverage
        coverage = self._compute_feature_coverage(text, heatmap)
        
        # 4. Keyword precision and recall
        precision, recall = self._compute_keyword_metrics(text, saliency_map)
        
        return AlignmentMetrics(
            text_saliency_overlap=text_saliency,
            spatial_consistency=spatial_score,
            feature_coverage=coverage,
            keyword_precision=precision,
            keyword_recall=recall
        )
    
    def _compute_text_saliency_overlap(self, text: str, heatmap: np.ndarray,
                                       image: np.ndarray) -> float:
        """Compute overlap between text mentions and salient regions"""
        h, w = heatmap.shape
        
        # Extract color mentions and check if they appear in salient regions
        color_score = 0
        color_count = 0
        
        for color in self.visual_vocabulary['color']:
            if color in text:
                color_count += 1
                # Check if color is present in salient regions
                salient_mask = heatmap > 0.5
                if salient_mask.any():
                    salient_region = image[salient_mask]
                    if len(salient_region) > 0:
                        # Simple color matching
                        color_present = self._check_color_in_region(color, salient_region)
                        color_score += color_present
        
        if color_count > 0:
            return color_score / color_count
        
        # Fallback: check if explanation mentions features proportional to saliency
        return float(heatmap.max() > 0.3)  # Basic check
    
    def _check_color_in_region(self, color: str, region: np.ndarray) -> float:
        """Check if a color is present in image region"""
        # Simple color matching based on mean RGB
        mean_color = region.mean(axis=0) if len(region.shape) > 1 else region
        
        color_map = {
            'red': [200, 50, 50],
            'blue': [50, 50, 200],
            'green': [50, 200, 50],
            'yellow': [200, 200, 50],
            'orange': [255, 165, 0],
            'white': [240, 240, 240],
            'black': [20, 20, 20],
            'brown': [139, 69, 19],
            'gray': [128, 128, 128],
            'grey': [128, 128, 128],
        }
        
        if color in color_map:
            target = np.array(color_map[color])
            distance = np.linalg.norm(mean_color[:3] - target)
            return 1.0 if distance < 100 else 0.0
        
        return 0.5  # Unknown color
    
    def _evaluate_spatial_references(self, text: str, heatmap: np.ndarray) -> float:
        """Evaluate consistency of spatial references with saliency"""
        h, w = heatmap.shape
        
        # Find salient region center
        if heatmap.max() < 0.1:
            return 0.5  # No clear saliency
        
        # Get center of mass of saliency
        y_coords, x_coords = np.where(heatmap > 0.5)
        if len(y_coords) == 0:
            return 0.5
        
        center_y = y_coords.mean() / h
        center_x = x_coords.mean() / w
        
        # Check spatial references in text
        scores = []
        
        if 'left' in text:
            scores.append(1.0 if center_x < 0.4 else 0.0)
        if 'right' in text:
            scores.append(1.0 if center_x > 0.6 else 0.0)
        if 'top' in text or 'upper' in text:
            scores.append(1.0 if center_y < 0.4 else 0.0)
        if 'bottom' in text or 'lower' in text:
            scores.append(1.0 if center_y > 0.6 else 0.0)
        if 'center' in text or 'middle' in text:
            scores.append(1.0 if 0.3 < center_x < 0.7 and 0.3 < center_y < 0.7 else 0.0)
        
        if scores:
            return np.mean(scores)
        return 0.5  # No spatial references to evaluate
    
    def _compute_feature_coverage(self, text: str, heatmap: np.ndarray) -> float:
        """Compute how well explanation covers salient features"""
        # Count visual feature mentions
        feature_count = 0
        
        for category, words in self.visual_vocabulary.items():
            for word in words:
                if word in text:
                    feature_count += 1
        
        # More features mentioned = better coverage (up to a point)
        # Normalize by saliency strength
        saliency_strength = heatmap.max()
        
        # Expect more features for stronger saliency
        expected_features = 2 + 3 * saliency_strength
        coverage = min(1.0, feature_count / expected_features)
        
        return coverage
    
    def _compute_keyword_metrics(self, text: str, saliency_map) -> Tuple[float, float]:
        """Compute precision and recall of visual keywords"""
        # Keywords mentioned in text
        mentioned = set()
        for category, words in self.visual_vocabulary.items():
            for word in words:
                if word in text:
                    mentioned.add(word)
        
        # "True" keywords based on class
        class_keywords = self._get_class_keywords(saliency_map.class_name)
        
        if not class_keywords:
            return 0.5, 0.5
        
        # Precision: what fraction of mentioned keywords are relevant
        if mentioned:
            precision = len(mentioned & class_keywords) / len(mentioned)
        else:
            precision = 0.0
        
        # Recall: what fraction of relevant keywords are mentioned
        recall = len(mentioned & class_keywords) / len(class_keywords)
        
        return precision, recall
    
    def _get_class_keywords(self, class_name: str) -> set:
        """Get expected visual keywords for a class"""
        class_keywords = {
            'person': {'face', 'body', 'arms', 'legs', 'head', 'tall'},
            'car': {'wheel', 'window', 'door', 'metal', 'shiny', 'rectangular'},
            'dog': {'fur', 'furry', 'legs', 'tail', 'ears', 'nose', 'brown'},
            'cat': {'fur', 'furry', 'ears', 'tail', 'eyes', 'whiskers'},
            'bird': {'feathers', 'wing', 'beak', 'small'},
            'bicycle': {'wheel', 'round', 'metal', 'thin'},
            'chair': {'legs', 'back', 'seat', 'wooden'},
            'bottle': {'tall', 'round', 'glass', 'transparent'},
        }
        
        return class_keywords.get(class_name.lower(), set())
    
    def evaluate_faithfulness(self,
                             image: np.ndarray,
                             detection,
                             saliency_map,
                             explanation,
                             detector) -> FaithfulnessMetrics:
        """
        Evaluate faithfulness of explanation via occlusion testing
        
        Args:
            image: Original image
            detection: Detection object
            saliency_map: SaliencyMap object
            explanation: Explanation object
            detector: YOLODetector for re-running inference
            
        Returns:
            FaithfulnessMetrics object
        """
        # 1. Occlusion sensitivity test
        occlusion_score = self._occlusion_test(
            image, detection, saliency_map.heatmap, detector
        )
        
        # 2. Counterfactual consistency
        counterfactual_score = self._counterfactual_test(
            image, detection, saliency_map.heatmap, detector
        )
        
        # 3. Feature attribution agreement
        attribution_score = self._feature_attribution_agreement(
            explanation, saliency_map
        )
        
        # 4. Prediction alignment
        prediction_score = self._prediction_alignment(explanation, detection)
        
        return FaithfulnessMetrics(
            occlusion_sensitivity=occlusion_score,
            counterfactual_consistency=counterfactual_score,
            feature_attribution_agreement=attribution_score,
            prediction_alignment=prediction_score
        )
    
    def _occlusion_test(self, image: np.ndarray, detection,
                       heatmap: np.ndarray, detector) -> float:
        """
        Test if occluding salient regions drops confidence
        
        Higher score = more faithful (saliency matches importance)
        """
        h, w = image.shape[:2]
        original_conf = detection.confidence
        
        # Get highly salient region mask
        heatmap_resized = cv2.resize(heatmap, (w, h))
        salient_mask = heatmap_resized > 0.5
        
        if not salient_mask.any():
            return 0.5  # No clear salient region
        
        # Occlude salient regions
        occluded_image = image.copy()
        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
        occluded_image[salient_mask] = mean_color
        
        # Re-run detection
        try:
            result = detector.detect(occluded_image, extract_regions=False)
            
            # Find matching detection
            x1, y1, x2, y2 = detection.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            matching_conf = 0.0
            for det in result.detections:
                if det.class_name == detection.class_name:
                    dx1, dy1, dx2, dy2 = det.bbox
                    det_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                    
                    # Check if centers are close
                    dist = np.sqrt((center[0] - det_center[0])**2 + 
                                  (center[1] - det_center[1])**2)
                    if dist < 50:  # Within 50 pixels
                        matching_conf = det.confidence
                        break
            
            # Confidence drop indicates faithful saliency
            conf_drop = original_conf - matching_conf
            
            # Normalize: expect significant drop for faithful explanations
            occlusion_score = min(1.0, conf_drop / 0.3)  # 30% drop = perfect score
            
            return max(0.0, occlusion_score)
            
        except Exception as e:
            print(f"Occlusion test error: {e}")
            return 0.5
    
    def _counterfactual_test(self, image: np.ndarray, detection,
                            heatmap: np.ndarray, detector) -> float:
        """
        Test counterfactual: occluding non-salient regions should not drop confidence
        """
        h, w = image.shape[:2]
        original_conf = detection.confidence
        
        # Get non-salient region mask
        heatmap_resized = cv2.resize(heatmap, (w, h))
        non_salient_mask = heatmap_resized < 0.3
        
        if not non_salient_mask.any():
            return 0.5
        
        # Occlude non-salient regions
        occluded_image = image.copy()
        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
        occluded_image[non_salient_mask] = mean_color
        
        try:
            result = detector.detect(occluded_image, extract_regions=False)
            
            # Find matching detection
            x1, y1, x2, y2 = detection.bbox
            
            matching_conf = 0.0
            for det in result.detections:
                if det.class_name == detection.class_name:
                    dx1, dy1, dx2, dy2 = det.bbox
                    # Check for overlap
                    iou = self._compute_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
                    if iou > 0.5:
                        matching_conf = det.confidence
                        break
            
            # Small confidence drop = good counterfactual consistency
            conf_drop = original_conf - matching_conf
            
            # Expect minimal drop
            if conf_drop < 0.1:
                return 1.0
            elif conf_drop < 0.2:
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            return 0.5
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _feature_attribution_agreement(self, explanation, saliency_map) -> float:
        """Check if explanation features match saliency attribution"""
        # Extract features mentioned in explanation
        mentioned_features = set(explanation.visual_features)
        
        # Check if features correspond to salient areas
        # Higher saliency in bbox = more agreement expected
        bbox_saliency = self._get_bbox_saliency(saliency_map)
        
        # More features + higher saliency = higher agreement
        feature_count = len(mentioned_features)
        
        if feature_count == 0:
            return 0.3  # No features mentioned
        
        # Combine feature count and saliency strength
        agreement = min(1.0, 0.3 * feature_count + 0.7 * bbox_saliency)
        
        return agreement
    
    def _get_bbox_saliency(self, saliency_map) -> float:
        """Get average saliency within bounding box"""
        x1, y1, x2, y2 = saliency_map.bbox
        heatmap = saliency_map.heatmap
        
        h, w = heatmap.shape
        
        # Ensure bbox is within heatmap bounds
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return 0.5
        
        bbox_region = heatmap[y1:y2, x1:x2]
        return float(bbox_region.mean())
    
    def _prediction_alignment(self, explanation, detection) -> float:
        """Check if explanation aligns with prediction"""
        text = explanation.explanation_text.lower()
        class_name = detection.class_name.lower()
        
        # Check if class is mentioned
        if class_name in text:
            alignment = 0.8
        else:
            alignment = 0.3
        
        # Check for uncertainty language
        uncertainty_phrases = ['might', 'could', 'possibly', 'uncertain', 'unclear']
        has_uncertainty = any(phrase in text for phrase in uncertainty_phrases)
        
        # Low confidence + uncertainty = good alignment
        # High confidence + no uncertainty = good alignment
        if detection.confidence > 0.7:
            if not has_uncertainty:
                alignment += 0.2
        else:
            if has_uncertainty:
                alignment += 0.2
        
        return min(1.0, alignment)
    
    def evaluate_quality(self, explanation) -> ExplanationQuality:
        """
        Evaluate text quality of explanation
        
        Args:
            explanation: Explanation object
            
        Returns:
            ExplanationQuality object
        """
        text = explanation.explanation_text
        
        # 1. Specificity: more specific visual features = higher score
        specificity = self._evaluate_specificity(text)
        
        # 2. Conciseness: shorter explanations with same info = higher score
        conciseness = self._evaluate_conciseness(text)
        
        # 3. Coherence: logical structure
        coherence = self._evaluate_coherence(text)
        
        # 4. Grounding: references to visual evidence
        grounding = self._evaluate_grounding(text)
        
        return ExplanationQuality(
            specificity=specificity,
            conciseness=conciseness,
            coherence=coherence,
            grounding=grounding
        )
    
    def _evaluate_specificity(self, text: str) -> float:
        """Evaluate specificity of explanation"""
        # Count specific visual features
        specific_count = 0
        
        for category, words in self.visual_vocabulary.items():
            for word in words:
                if word in text.lower():
                    specific_count += 1
        
        # More specific features = higher score
        return min(1.0, specific_count / 5)
    
    def _evaluate_conciseness(self, text: str) -> float:
        """Evaluate conciseness of explanation"""
        words = text.split()
        word_count = len(words)
        
        # Ideal length: 30-60 words
        if 30 <= word_count <= 60:
            return 1.0
        elif word_count < 30:
            return word_count / 30
        else:
            return max(0.3, 60 / word_count)
    
    def _evaluate_coherence(self, text: str) -> float:
        """Evaluate logical coherence of explanation"""
        # Check for sentence structure
        sentences = text.split('.')
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) == 0:
            return 0.3
        
        # Check for logical connectors
        connectors = ['because', 'since', 'therefore', 'however', 'also', 
                     'furthermore', 'additionally', 'which', 'that']
        connector_count = sum(1 for c in connectors if c in text.lower())
        
        coherence = min(1.0, 0.5 + 0.1 * connector_count + 0.1 * len(valid_sentences))
        
        return coherence
    
    def _evaluate_grounding(self, text: str) -> float:
        """Evaluate visual grounding in explanation"""
        # Check for phrases indicating visual observation
        grounding_phrases = [
            'can see', 'visible', 'appears', 'shows', 'displays',
            'the image', 'in the image', 'looking at', 'observe',
            'detected', 'identified', 'features', 'characteristics'
        ]
        
        grounding_count = sum(1 for phrase in grounding_phrases 
                            if phrase in text.lower())
        
        return min(1.0, grounding_count / 3)
    
    def evaluate_detection(self,
                          image: np.ndarray,
                          detection,
                          saliency_map,
                          explanation,
                          detector=None) -> EvaluationResult:
        """
        Complete evaluation for a single detection
        
        Args:
            image: Original image
            detection: Detection object
            saliency_map: SaliencyMap object
            explanation: Explanation object
            detector: Optional detector for faithfulness tests
            
        Returns:
            EvaluationResult object
        """
        # Alignment evaluation
        alignment = self.evaluate_alignment(explanation, saliency_map, image)
        
        # Quality evaluation
        quality = self.evaluate_quality(explanation)
        
        # Faithfulness evaluation (if detector provided)
        if detector is not None:
            faithfulness = self.evaluate_faithfulness(
                image, detection, saliency_map, explanation, detector
            )
        else:
            faithfulness = FaithfulnessMetrics(
                occlusion_sensitivity=0.5,
                counterfactual_consistency=0.5,
                feature_attribution_agreement=0.5,
                prediction_alignment=self._prediction_alignment(explanation, detection)
            )
        
        return EvaluationResult(
            detection_id=detection.bbox[0],  # Use x1 as ID for now
            class_name=detection.class_name,
            alignment=alignment,
            faithfulness=faithfulness,
            quality=quality
        )


class EvaluationReport:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self, results: List[EvaluationResult]):
        """
        Initialize report generator
        
        Args:
            results: List of EvaluationResult objects
        """
        self.results = results
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.results:
            return {'error': 'No results to summarize'}
        
        # Aggregate metrics
        alignment_scores = [r.alignment.to_dict()['mean_alignment'] for r in self.results]
        faithfulness_scores = [r.faithfulness.to_dict()['mean_faithfulness'] for r in self.results]
        quality_scores = [r.quality.to_dict()['mean_quality'] for r in self.results]
        overall_scores = [r.get_overall_score() for r in self.results]
        
        # By class
        class_scores = defaultdict(list)
        for r in self.results:
            class_scores[r.class_name].append(r.get_overall_score())
        
        return {
            'num_evaluations': len(self.results),
            'alignment': {
                'mean': np.mean(alignment_scores),
                'std': np.std(alignment_scores),
                'min': np.min(alignment_scores),
                'max': np.max(alignment_scores)
            },
            'faithfulness': {
                'mean': np.mean(faithfulness_scores),
                'std': np.std(faithfulness_scores),
                'min': np.min(faithfulness_scores),
                'max': np.max(faithfulness_scores)
            },
            'quality': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            },
            'overall': {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'min': np.min(overall_scores),
                'max': np.max(overall_scores)
            },
            'by_class': {cls: np.mean(scores) for cls, scores in class_scores.items()}
        }
    
    def get_failure_cases(self, threshold: float = 0.4) -> List[EvaluationResult]:
        """Get results with low overall scores"""
        return [r for r in self.results if r.get_overall_score() < threshold]
    
    def get_success_cases(self, threshold: float = 0.7) -> List[EvaluationResult]:
        """Get results with high overall scores"""
        return [r for r in self.results if r.get_overall_score() >= threshold]
    
    def save_report(self, filepath: str):
        """Save report to JSON file"""
        report = {
            'summary': self.generate_summary(),
            'detailed_results': [r.to_dict() for r in self.results],
            'failure_cases': [r.to_dict() for r in self.get_failure_cases()],
            'success_cases': [r.to_dict() for r in self.get_success_cases()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filepath}")


def run_demo():
    """Run evaluation demo"""
    print("Evaluation Module Demo")
    print("=" * 50)
    
    # Create mock data for testing
    from dataclasses import dataclass as dc
    
    @dc
    class MockExplanation:
        explanation_text: str = "This appears to be a dog with brown fur, four legs visible, and a wagging tail."
        visual_features: List[str] = None
        
        def __post_init__(self):
            self.visual_features = ['brown fur', 'four legs', 'tail']
    
    @dc
    class MockSaliencyMap:
        heatmap: np.ndarray = None
        bbox: Tuple[int, int, int, int] = (50, 50, 200, 200)
        class_name: str = "dog"
        
        def __post_init__(self):
            self.heatmap = np.random.rand(224, 224)
    
    @dc
    class MockDetection:
        class_name: str = "dog"
        confidence: float = 0.85
        bbox: Tuple[int, int, int, int] = (50, 50, 200, 200)
    
    # Initialize evaluator
    evaluator = ExplanationEvaluator()
    
    # Create mock objects
    explanation = MockExplanation()
    saliency = MockSaliencyMap()
    detection = MockDetection()
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Evaluate alignment
    alignment = evaluator.evaluate_alignment(explanation, saliency, image)
    print("\nAlignment Metrics:")
    for k, v in alignment.to_dict().items():
        print(f"  {k}: {v:.3f}")
    
    # Evaluate quality
    quality = evaluator.evaluate_quality(explanation)
    print("\nQuality Metrics:")
    for k, v in quality.to_dict().items():
        print(f"  {k}: {v:.3f}")
    
    # Full evaluation
    result = evaluator.evaluate_detection(image, detection, saliency, explanation)
    print(f"\nOverall Score: {result.get_overall_score():.3f}")


if __name__ == "__main__":
    run_demo()

