"""
CLIP Embedding Module for Semantic Similarity Analysis
Computes visual-text semantic grounding for detected regions
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import clip

from config import CLIPConfig, COCO_CLASSES, get_clip_prompts


@dataclass
class SemanticMatch:
    """Result of semantic similarity matching"""
    predicted_class: str
    predicted_confidence: float  # From YOLO
    clip_scores: Dict[str, float]  # Class name -> CLIP similarity
    top_k_classes: List[Tuple[str, float]]  # Top-k most similar classes
    alignment_score: float  # Agreement between YOLO and CLIP
    
    def __repr__(self):
        return (f"SemanticMatch(pred={self.predicted_class}, "
                f"alignment={self.alignment_score:.3f}, "
                f"top_clip={self.top_k_classes[0] if self.top_k_classes else 'N/A'})")


@dataclass
class RegionEmbedding:
    """CLIP embedding for a detected region"""
    region_id: int
    image_embedding: torch.Tensor
    class_name: str
    confidence: float
    semantic_match: Optional[SemanticMatch] = None


class CLIPEmbedder:
    """CLIP-based semantic analysis for object detection"""
    
    def __init__(self, config: Optional[CLIPConfig] = None):
        """
        Initialize CLIP embedder
        
        Args:
            config: CLIPConfig with model settings
        """
        self.config = config or CLIPConfig()
        self.device = self._setup_device()
        self.model, self.preprocess = self._load_model()
        self.class_embeddings = None  # Cached text embeddings
        
    def _setup_device(self) -> str:
        """Setup computation device"""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load CLIP model"""
        print(f"Loading CLIP model: {self.config.model_name}")
        model, preprocess = clip.load(self.config.model_name, device=self.device)
        model.eval()
        return model, preprocess
    
    @torch.no_grad()
    def encode_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Encode image to CLIP embedding
        
        Args:
            image: Input image (numpy array, PIL Image, or tensor)
            
        Returns:
            Normalized image embedding
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        embedding = self.model.encode_image(image)
        embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    @torch.no_grad()
    def encode_images_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Encode batch of images
        
        Args:
            images: List of images
            
        Returns:
            Batch of normalized embeddings
        """
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed.append(self.preprocess(img))
        
        batch = torch.stack(processed).to(self.device)
        embeddings = self.model.encode_image(batch)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to CLIP embedding
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Normalized text embedding(s)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        embeddings = self.model.encode_text(tokens)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    def compute_class_embeddings(self, class_names: Optional[List[str]] = None,
                                  use_prompts: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute and cache text embeddings for all classes
        
        Args:
            class_names: List of class names (defaults to COCO classes)
            use_prompts: Use multiple prompts per class and average
            
        Returns:
            Dictionary mapping class names to embeddings
        """
        if class_names is None:
            class_names = COCO_CLASSES
        
        class_embeddings = {}
        
        for class_name in class_names:
            if use_prompts:
                # Use multiple prompts and average
                prompts = get_clip_prompts(class_name)
                embeddings = self.encode_text(prompts)
                avg_embedding = embeddings.mean(dim=0, keepdim=True)
                avg_embedding = F.normalize(avg_embedding, dim=-1)
                class_embeddings[class_name] = avg_embedding
            else:
                # Single prompt
                text = f"a photo of a {class_name}"
                class_embeddings[class_name] = self.encode_text(text)
        
        self.class_embeddings = class_embeddings
        return class_embeddings
    
    def compute_semantic_similarity(self, 
                                   image: Union[np.ndarray, Image.Image],
                                   predicted_class: str,
                                   confidence: float,
                                   top_k: int = 5) -> SemanticMatch:
        """
        Compute semantic similarity between image region and class labels
        
        Args:
            image: Cropped image region
            predicted_class: YOLO predicted class name
            confidence: YOLO confidence score
            top_k: Number of top classes to return
            
        Returns:
            SemanticMatch object with similarity scores
        """
        # Ensure class embeddings are computed
        if self.class_embeddings is None:
            self.compute_class_embeddings()
        
        # Encode image
        image_embedding = self.encode_image(image)
        
        # Compute similarity with all classes
        clip_scores = {}
        for class_name, text_embedding in self.class_embeddings.items():
            similarity = (image_embedding @ text_embedding.T).item()
            clip_scores[class_name] = similarity
        
        # Get top-k classes
        sorted_classes = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_classes = sorted_classes[:top_k]
        
        # Compute alignment score (how well YOLO and CLIP agree)
        predicted_rank = next(
            (i for i, (name, _) in enumerate(sorted_classes) if name == predicted_class),
            len(sorted_classes)
        )
        
        # Alignment based on rank and score difference
        predicted_score = clip_scores.get(predicted_class, 0)
        top_score = top_k_classes[0][1] if top_k_classes else 0
        
        # Alignment is high if predicted class is top-ranked or has similar score to top
        rank_alignment = 1.0 / (1.0 + predicted_rank)
        score_alignment = predicted_score / (top_score + 1e-6) if top_score > 0 else 0
        alignment_score = 0.5 * rank_alignment + 0.5 * score_alignment
        
        return SemanticMatch(
            predicted_class=predicted_class,
            predicted_confidence=confidence,
            clip_scores=clip_scores,
            top_k_classes=top_k_classes,
            alignment_score=alignment_score
        )
    
    def analyze_detections(self, detections, top_k: int = 5) -> List[RegionEmbedding]:
        """
        Analyze all detections from YOLO
        
        Args:
            detections: List of Detection objects from YOLODetector
            top_k: Number of top classes to return per detection
            
        Returns:
            List of RegionEmbedding objects with semantic analysis
        """
        region_embeddings = []
        
        # Ensure class embeddings are computed
        if self.class_embeddings is None:
            self.compute_class_embeddings()
        
        for i, det in enumerate(detections):
            if det.region is None:
                continue
            
            # Get image embedding
            image_embedding = self.encode_image(det.region)
            
            # Compute semantic match
            semantic_match = self.compute_semantic_similarity(
                det.region,
                det.class_name,
                det.confidence,
                top_k
            )
            
            region_embedding = RegionEmbedding(
                region_id=i,
                image_embedding=image_embedding.cpu(),
                class_name=det.class_name,
                confidence=det.confidence,
                semantic_match=semantic_match
            )
            region_embeddings.append(region_embedding)
        
        return region_embeddings
    
    def compute_region_similarity_matrix(self, 
                                        regions: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise similarity between all regions
        
        Args:
            regions: List of cropped image regions
            
        Returns:
            NxN similarity matrix
        """
        embeddings = self.encode_images_batch(regions)
        similarity_matrix = (embeddings @ embeddings.T).cpu().numpy()
        return similarity_matrix
    
    def find_similar_regions(self, query_region: np.ndarray,
                            candidate_regions: List[np.ndarray],
                            threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        Find regions similar to query
        
        Args:
            query_region: Query image region
            candidate_regions: List of candidate regions
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity) tuples for similar regions
        """
        query_embedding = self.encode_image(query_region)
        
        similar = []
        for i, region in enumerate(candidate_regions):
            region_embedding = self.encode_image(region)
            similarity = (query_embedding @ region_embedding.T).item()
            
            if similarity >= threshold:
                similar.append((i, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def explain_classification_mismatch(self, 
                                       region: np.ndarray,
                                       yolo_class: str,
                                       clip_top_class: str) -> Dict:
        """
        Analyze why YOLO and CLIP might disagree
        
        Args:
            region: Image region
            yolo_class: YOLO predicted class
            clip_top_class: CLIP top predicted class
            
        Returns:
            Analysis dictionary
        """
        # Get embeddings for both classes
        yolo_text = f"a photo of a {yolo_class}"
        clip_text = f"a photo of a {clip_top_class}"
        
        image_embedding = self.encode_image(region)
        yolo_embedding = self.encode_text(yolo_text)
        clip_embedding = self.encode_text(clip_top_class)
        
        yolo_sim = (image_embedding @ yolo_embedding.T).item()
        clip_sim = (image_embedding @ clip_embedding.T).item()
        
        # Get similarity with related classes
        related_to_yolo = self._get_semantically_related(yolo_class)
        related_to_clip = self._get_semantically_related(clip_top_class)
        
        return {
            'yolo_class': yolo_class,
            'yolo_similarity': yolo_sim,
            'clip_class': clip_top_class,
            'clip_similarity': clip_sim,
            'similarity_gap': clip_sim - yolo_sim,
            'related_to_yolo': related_to_yolo,
            'related_to_clip': related_to_clip,
            'potential_confusion': yolo_class in related_to_clip or clip_top_class in related_to_yolo
        }
    
    def _get_semantically_related(self, class_name: str, top_k: int = 3) -> List[str]:
        """Get semantically related classes based on embedding similarity"""
        if self.class_embeddings is None:
            self.compute_class_embeddings()
        
        target_embedding = self.class_embeddings.get(class_name)
        if target_embedding is None:
            target_embedding = self.encode_text(f"a photo of a {class_name}")
        
        similarities = []
        for name, embedding in self.class_embeddings.items():
            if name != class_name:
                sim = (target_embedding @ embedding.T).item()
                similarities.append((name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:top_k]]
    
    def generate_semantic_report(self, region_embeddings: List[RegionEmbedding]) -> Dict:
        """
        Generate comprehensive semantic analysis report
        
        Args:
            region_embeddings: List of analyzed regions
            
        Returns:
            Report dictionary
        """
        report = {
            'total_regions': len(region_embeddings),
            'high_alignment': 0,
            'medium_alignment': 0,
            'low_alignment': 0,
            'class_distribution': {},
            'alignment_scores': [],
            'mismatches': []
        }
        
        for re in region_embeddings:
            if re.semantic_match is None:
                continue
            
            alignment = re.semantic_match.alignment_score
            report['alignment_scores'].append(alignment)
            
            if alignment > 0.7:
                report['high_alignment'] += 1
            elif alignment > 0.4:
                report['medium_alignment'] += 1
            else:
                report['low_alignment'] += 1
            
            # Track class distribution
            cls = re.class_name
            report['class_distribution'][cls] = report['class_distribution'].get(cls, 0) + 1
            
            # Track mismatches
            if re.semantic_match.top_k_classes:
                top_clip_class = re.semantic_match.top_k_classes[0][0]
                if top_clip_class != re.class_name:
                    report['mismatches'].append({
                        'region_id': re.region_id,
                        'yolo_class': re.class_name,
                        'clip_class': top_clip_class,
                        'alignment': alignment
                    })
        
        # Summary statistics
        if report['alignment_scores']:
            report['mean_alignment'] = np.mean(report['alignment_scores'])
            report['std_alignment'] = np.std(report['alignment_scores'])
        
        return report


def run_demo():
    """Run demo of CLIP embedder"""
    import matplotlib.pyplot as plt
    
    # Initialize embedder
    config = CLIPConfig(model_name="ViT-B/32")
    embedder = CLIPEmbedder(config)
    
    # Compute class embeddings
    print("Computing class embeddings...")
    embedder.compute_class_embeddings()
    
    # Create a simple test image
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    test_img[50:150, 50:150] = [255, 128, 0]  # Orange square
    
    # Analyze
    match = embedder.compute_semantic_similarity(test_img, "cat", 0.9)
    
    print("\nSemantic Match Results:")
    print(f"  Predicted: {match.predicted_class} (conf: {match.predicted_confidence:.2f})")
    print(f"  Alignment: {match.alignment_score:.3f}")
    print(f"  Top CLIP matches:")
    for cls, score in match.top_k_classes[:5]:
        marker = " <- YOLO" if cls == match.predicted_class else ""
        print(f"    {cls}: {score:.3f}{marker}")


if __name__ == "__main__":
    run_demo()

