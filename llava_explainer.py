"""
LLaVA Explanation Module
Generates natural language explanations for object detection predictions
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

from config import (
    LLaVAConfig, 
    LLAVA_SYSTEM_PROMPT, 
    LLAVA_EXPLANATION_TEMPLATE,
    LLAVA_VERIFICATION_TEMPLATE
)


@dataclass
class Explanation:
    """Natural language explanation for a detection"""
    detection_id: int
    class_name: str
    confidence: float
    explanation_text: str
    visual_features: List[str]  # Extracted visual features from explanation
    supporting_evidence: str
    uncertainty_indicators: List[str]
    prompt_used: str
    
    def __repr__(self):
        return f"Explanation({self.class_name}: '{self.explanation_text[:50]}...')"
    
    def get_feature_keywords(self) -> List[str]:
        """Extract key visual feature keywords from explanation"""
        # Common visual feature patterns
        patterns = [
            r'\b(color|shape|size|texture|pattern)\b',
            r'\b(round|square|rectangular|elongated|curved)\b',
            r'\b(red|blue|green|yellow|black|white|brown|gray|orange|pink|purple)\b',
            r'\b(fur|feathers|scales|skin|metal|wood|plastic|glass)\b',
            r'\b(striped|spotted|solid|checkered|dotted)\b',
            r'\b(large|small|medium|tiny|huge|long|short|tall|wide|narrow)\b'
        ]
        
        keywords = []
        text_lower = self.explanation_text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        return list(set(keywords))


class LLaVAExplainer:
    """
    LLaVA-based explanation generator for object detections
    
    Uses vision-language model to generate natural language explanations
    describing why an object was classified in a particular way.
    """
    
    def __init__(self, config: Optional[LLaVAConfig] = None):
        """
        Initialize LLaVA explainer
        
        Args:
            config: LLaVAConfig with model settings
        """
        self.config = config or LLaVAConfig()
        self.device = self._setup_device()
        self.model = None
        self.processor = None
        self._load_model()
        
    def _setup_device(self) -> str:
        """Setup computation device"""
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load LLaVA model and processor"""
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        
        print(f"Loading LLaVA model: {self.config.model_name}")
        
        # Configure quantization for memory efficiency
        if self.config.load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model.eval()
        
        print("LLaVA model loaded successfully")
    
    def _create_prompt(self, class_name: str, confidence: float, 
                      prompt_type: str = "explanation") -> str:
        """
        Create prompt for LLaVA
        
        Args:
            class_name: Detected class name
            confidence: Detection confidence
            prompt_type: Type of prompt ('explanation', 'verification', 'detailed')
            
        Returns:
            Formatted prompt string
        """
        if prompt_type == "explanation":
            prompt = LLAVA_EXPLANATION_TEMPLATE.format(
                class_name=class_name,
                confidence=confidence
            )
        elif prompt_type == "verification":
            prompt = LLAVA_VERIFICATION_TEMPLATE.format(class_name=class_name)
        elif prompt_type == "detailed":
            prompt = f"""Analyze this image region carefully. A detection model identified this as: {class_name} (confidence: {confidence:.1%})

Please provide:
1. A list of 3-5 specific visual features you can see that support or contradict this classification
2. Your assessment of whether this classification appears correct
3. Any alternative objects this could potentially be

Be specific and describe only what you can actually observe in this image."""
        else:
            prompt = f"What object do you see in this image? The model predicted: {class_name}"
        
        return prompt
    
    def _extract_visual_features(self, explanation: str) -> List[str]:
        """Extract mentioned visual features from explanation text"""
        features = []
        
        # Feature extraction patterns
        feature_patterns = [
            r'(?:has|with|featuring|shows?|contains?|displays?)\s+(?:a\s+)?([^,.]+(?:shape|color|texture|pattern|size))',
            r'(?:the\s+)?(\w+\s+(?:fur|feathers|scales|skin|surface|body|head|tail|legs?|wings?))',
            r'((?:round|square|oval|elongated|curved|straight)\s+\w+)',
            r'((?:bright|dark|light|deep|pale)\s+\w+\s+color)',
            r'(\w+(?:ed|ing)\s+(?:pattern|texture|surface|appearance))',
        ]
        
        explanation_lower = explanation.lower()
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, explanation_lower)
            features.extend(matches)
        
        # Clean up features
        features = [f.strip() for f in features if len(f.strip()) > 3]
        return list(set(features))[:10]  # Limit to top 10
    
    def _extract_uncertainty(self, explanation: str) -> List[str]:
        """Extract uncertainty indicators from explanation"""
        uncertainty_phrases = [
            'might be', 'could be', 'appears to be', 'seems like',
            'possibly', 'potentially', 'uncertain', 'unclear',
            'difficult to determine', 'hard to tell', 'not sure',
            'may be', 'looks like it could', 'resembles'
        ]
        
        found = []
        explanation_lower = explanation.lower()
        
        for phrase in uncertainty_phrases:
            if phrase in explanation_lower:
                found.append(phrase)
        
        return found
    
    @torch.no_grad()
    def generate_explanation(self, 
                            image: Union[np.ndarray, Image.Image],
                            class_name: str,
                            confidence: float,
                            prompt_type: str = "explanation") -> Explanation:
        """
        Generate explanation for a detected object
        
        Args:
            image: Cropped image region of detected object
            class_name: Predicted class name
            confidence: Detection confidence
            prompt_type: Type of prompt to use
            
        Returns:
            Explanation object
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Create prompt
        prompt = self._create_prompt(class_name, confidence, prompt_type)
        
        # Format for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )
        
        # Decode response
        generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the assistant's response (after the prompt)
        if "assistant" in generated_text.lower():
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text.split(prompt)[-1].strip()
        
        # Clean up response
        response = response.strip()
        if response.startswith(":"):
            response = response[1:].strip()
        
        # Extract features and uncertainty
        visual_features = self._extract_visual_features(response)
        uncertainty = self._extract_uncertainty(response)
        
        # Extract supporting evidence (first sentence or two)
        sentences = response.split('.')
        supporting = '. '.join(sentences[:2]).strip() + '.' if sentences else response
        
        return Explanation(
            detection_id=-1,  # Will be set by caller
            class_name=class_name,
            confidence=confidence,
            explanation_text=response,
            visual_features=visual_features,
            supporting_evidence=supporting,
            uncertainty_indicators=uncertainty,
            prompt_used=prompt
        )
    
    def explain_detections(self, 
                          detections,
                          prompt_type: str = "explanation",
                          max_detections: int = 10) -> List[Explanation]:
        """
        Generate explanations for multiple detections
        
        Args:
            detections: List of Detection objects from YOLODetector
            prompt_type: Type of prompt to use
            max_detections: Maximum number of detections to explain
            
        Returns:
            List of Explanation objects
        """
        explanations = []
        
        for i, det in enumerate(detections[:max_detections]):
            if det.region is None:
                continue
            
            try:
                explanation = self.generate_explanation(
                    det.region,
                    det.class_name,
                    det.confidence,
                    prompt_type
                )
                explanation.detection_id = i
                explanations.append(explanation)
                
                print(f"Generated explanation for detection {i}: {det.class_name}")
                
            except Exception as e:
                print(f"Warning: Could not generate explanation for detection {i}: {e}")
                continue
        
        return explanations
    
    def compare_explanations(self,
                            image: Union[np.ndarray, Image.Image],
                            yolo_class: str,
                            clip_class: str,
                            confidence: float) -> Dict:
        """
        Generate comparative explanation when YOLO and CLIP disagree
        
        Args:
            image: Image region
            yolo_class: YOLO predicted class
            clip_class: CLIP predicted class
            confidence: Detection confidence
            
        Returns:
            Dictionary with comparison results
        """
        prompt = f"""A detection model classified this image region as "{yolo_class}" with {confidence:.1%} confidence.
However, a semantic analysis suggests it might be a "{clip_class}" instead.

Please examine the image carefully and:
1. Describe the key visual features you observe
2. Explain which classification ('{yolo_class}' or '{clip_class}') better matches what you see
3. If there's ambiguity, explain why both could be reasonable interpretations"""
        
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Generate response
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # Analyze response to determine which class LLaVA favors
        yolo_mentions = response.lower().count(yolo_class.lower())
        clip_mentions = response.lower().count(clip_class.lower())
        
        # Look for preference indicators
        prefers_yolo = any(phrase in response.lower() for phrase in 
                         [f"is a {yolo_class.lower()}", f"appears to be a {yolo_class.lower()}",
                          f"correctly identified as {yolo_class.lower()}"])
        prefers_clip = any(phrase in response.lower() for phrase in
                         [f"is a {clip_class.lower()}", f"appears to be a {clip_class.lower()}",
                          f"actually a {clip_class.lower()}"])
        
        preferred = "yolo" if prefers_yolo else ("clip" if prefers_clip else "uncertain")
        
        return {
            'yolo_class': yolo_class,
            'clip_class': clip_class,
            'explanation': response,
            'preferred_classification': preferred,
            'yolo_mentions': yolo_mentions,
            'clip_mentions': clip_mentions,
            'visual_features': self._extract_visual_features(response)
        }
    
    def verify_detection(self, 
                        image: Union[np.ndarray, Image.Image],
                        class_name: str,
                        confidence: float) -> Dict:
        """
        Ask LLaVA to verify a detection without bias
        
        Args:
            image: Image region
            class_name: Detected class name
            confidence: Detection confidence
            
        Returns:
            Verification result dictionary
        """
        # First, ask what LLaVA sees without revealing the prediction
        blind_prompt = "What object do you see in this image? Describe it briefly."
        
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": blind_prompt}
                ]
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        blind_response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        if "assistant" in blind_response.lower():
            blind_response = blind_response.split("assistant")[-1].strip()
        
        # Check if LLaVA's blind prediction matches
        class_in_response = class_name.lower() in blind_response.lower()
        
        # Get synonyms and related terms
        related_terms = self._get_related_terms(class_name)
        any_related = any(term.lower() in blind_response.lower() for term in related_terms)
        
        return {
            'detected_class': class_name,
            'confidence': confidence,
            'llava_blind_response': blind_response,
            'class_mentioned': class_in_response,
            'related_term_mentioned': any_related,
            'agreement': class_in_response or any_related,
            'related_terms_found': [t for t in related_terms if t.lower() in blind_response.lower()]
        }
    
    def _get_related_terms(self, class_name: str) -> List[str]:
        """Get related terms for a class"""
        related = {
            'person': ['human', 'man', 'woman', 'people', 'individual'],
            'car': ['vehicle', 'automobile', 'sedan', 'SUV'],
            'dog': ['canine', 'puppy', 'hound'],
            'cat': ['feline', 'kitten'],
            'bird': ['avian', 'fowl'],
            'bicycle': ['bike', 'cycle'],
            'motorcycle': ['motorbike', 'scooter'],
            'airplane': ['plane', 'aircraft', 'jet'],
            'bus': ['coach', 'transit'],
            'train': ['locomotive', 'railway'],
            'truck': ['lorry', 'pickup'],
            'boat': ['ship', 'vessel', 'watercraft'],
            'horse': ['equine', 'stallion', 'mare'],
            'sheep': ['lamb', 'ewe'],
            'cow': ['cattle', 'bovine', 'bull'],
            'elephant': ['pachyderm'],
            'bear': ['grizzly', 'polar bear'],
            'zebra': ['equine', 'striped horse'],
            'giraffe': ['tall animal'],
        }
        return related.get(class_name.lower(), [class_name])


class SimpleLLMExplainer:
    """
    Simplified explainer that doesn't require LLaVA
    Uses rule-based explanations for demo/testing purposes
    """
    
    def __init__(self):
        self.class_descriptions = self._load_descriptions()
    
    def _load_descriptions(self) -> Dict[str, Dict]:
        """Load class-specific feature descriptions"""
        return {
            'person': {
                'features': ['human form', 'face', 'limbs', 'clothing'],
                'template': 'This appears to be a person based on the {feature}. The image shows human characteristics including {detail}.'
            },
            'car': {
                'features': ['wheels', 'windows', 'body shape', 'headlights'],
                'template': 'This is identified as a car due to its {feature}. Visible automotive features include {detail}.'
            },
            'dog': {
                'features': ['four legs', 'fur', 'snout', 'tail'],
                'template': 'The detection shows a dog, identifiable by its {feature}. Canine characteristics visible: {detail}.'
            },
            'cat': {
                'features': ['pointed ears', 'whiskers', 'fur', 'tail'],
                'template': 'This appears to be a cat based on {feature}. Feline features observed: {detail}.'
            },
            'default': {
                'features': ['shape', 'color', 'texture', 'size'],
                'template': 'The object is classified as {class_name} based on its {feature}.'
            }
        }
    
    def generate_explanation(self,
                            image: Union[np.ndarray, Image.Image],
                            class_name: str,
                            confidence: float) -> Explanation:
        """Generate rule-based explanation"""
        desc = self.class_descriptions.get(class_name.lower(), 
                                           self.class_descriptions['default'])
        
        feature = np.random.choice(desc['features'])
        detail = ', '.join(np.random.choice(desc['features'], size=min(2, len(desc['features'])), replace=False))
        
        text = desc['template'].format(
            class_name=class_name,
            feature=feature,
            detail=detail
        )
        
        if confidence < 0.5:
            text += f" However, with {confidence:.0%} confidence, there is some uncertainty in this classification."
        
        return Explanation(
            detection_id=-1,
            class_name=class_name,
            confidence=confidence,
            explanation_text=text,
            visual_features=desc['features'],
            supporting_evidence=text.split('.')[0] + '.',
            uncertainty_indicators=['some uncertainty'] if confidence < 0.5 else [],
            prompt_used='rule-based'
        )


def run_demo():
    """Run LLaVA explainer demo"""
    print("LLaVA Explainer Demo")
    print("=" * 50)
    
    # Try simple explainer first (no model download required)
    print("\nUsing SimpleLLMExplainer (rule-based):")
    simple = SimpleLLMExplainer()
    
    # Create dummy image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    explanation = simple.generate_explanation(test_img, "dog", 0.85)
    print(f"\nClass: {explanation.class_name}")
    print(f"Confidence: {explanation.confidence:.2%}")
    print(f"Explanation: {explanation.explanation_text}")
    print(f"Visual Features: {explanation.visual_features}")
    
    # Only try LLaVA if explicitly requested
    try_llava = False  # Set to True to test actual LLaVA
    
    if try_llava:
        print("\n\nUsing LLaVA model:")
        try:
            config = LLaVAConfig(load_in_4bit=True)
            explainer = LLaVAExplainer(config)
            
            explanation = explainer.generate_explanation(test_img, "dog", 0.85)
            print(f"\nExplanation: {explanation.explanation_text}")
        except Exception as e:
            print(f"Could not load LLaVA: {e}")


if __name__ == "__main__":
    run_demo()

