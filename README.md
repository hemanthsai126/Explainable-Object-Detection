# Explainable Object Detection Framework

A comprehensive framework for interpretable object detection that combines modern deep learning models with explainability techniques.

## Overview

This framework integrates:
- **YOLOv8** for object detection
- **CLIP** for visual-text semantic grounding  
- **LLaVA** for natural language explanations
- **Grad-CAM** for visual saliency visualization

For each detected object, the system:
1. Detects and localizes objects using YOLOv8
2. Computes CLIP embeddings to measure semantic similarity between regions and class labels
3. Generates Grad-CAM saliency maps to visualize influential image regions
4. Uses LLaVA to generate natural language explanations (optional)
5. Evaluates explanation quality using alignment and faithfulness metrics

## Installation

```bash
# Clone or download the project
cd CV

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Requirements
- **Minimum**: 8GB VRAM (without LLaVA)
- **Recommended**: 16GB+ VRAM (with LLaVA)
- CPU mode is supported but significantly slower

## Quick Start

### 🌐 Run Web Application (Recommended)
```bash
# Launch the Streamlit web app
streamlit run app.py

# Or use the launcher script
python run_app.py
```
Then open your browser to **http://localhost:8501**

### Run CLI Demo
```bash
python main.py --demo
```

### Process Single Image
```bash
python main.py --image path/to/image.jpg --output outputs/
```

### Process Directory
```bash
python main.py --image_dir path/to/images/ --output outputs/
```

### Run Evaluation on COCO
```bash
python main.py --evaluate --data_dir data/coco/ --num_samples 100
```

## Usage Options

```
Arguments:
  --image PATH          Process single image
  --image_dir PATH      Process directory of images
  --demo               Run demo with test image
  --evaluate           Run evaluation on COCO dataset
  
  --output, -o PATH    Output directory (default: outputs/)
  --save_json          Save results as JSON (default: True)
  --save_vis           Save visualizations (default: True)
  
  --yolo_model MODEL   YOLOv8 model (yolov8n/s/m/l/x.pt)
  --clip_model MODEL   CLIP model (ViT-B/32, ViT-B/16, etc.)
  --use_llava          Enable LLaVA explanations (GPU required)
  
  --max_detections N   Max detections per image (default: 10)
  --conf_threshold F   Detection confidence threshold (default: 0.25)
  --no_saliency        Skip Grad-CAM computation
  --no_eval            Skip evaluation metrics
  
  --cpu                Force CPU mode
  --device DEVICE      Device (auto/cuda/cpu)
```

## 🌐 Web Application

The Streamlit web app provides an interactive interface for:

- **📷 Image Upload**: Drag & drop or browse for images
- **⚙️ Configuration**: Adjust model settings in real-time
- **🔍 Detection**: View detected objects with bounding boxes
- **📊 Analysis**: Explore CLIP alignment and Grad-CAM saliency maps
- **💬 Explanations**: Read AI-generated explanations for each detection
- **📈 Evaluation**: Interactive charts showing quality metrics
- **📥 Export**: Download results as JSON

### Screenshots

After running `streamlit run app.py`, you'll see:
1. **Upload Tab**: Upload images and run detection
2. **Analysis Tab**: Detailed per-detection analysis
3. **Evaluation Tab**: Quality metrics with interactive charts
4. **About Tab**: System information and documentation

## Project Structure

```
CV/
├── app.py               # Streamlit web application
├── run_app.py           # App launcher script
├── main.py              # CLI entry point
├── config.py            # Configuration settings
├── pipeline.py          # Complete pipeline integration
├── yolo_detector.py     # YOLOv8 detection module
├── clip_embeddings.py   # CLIP semantic analysis
├── gradcam_yolo.py      # Grad-CAM saliency maps
├── llava_explainer.py   # LLaVA explanation generation
├── evaluation.py        # Evaluation metrics
├── visualization.py     # Visualization utilities
├── analysis_notebook.ipynb  # Jupyter notebook
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Module Details

### YOLOv8 Detector (`yolo_detector.py`)
- Runs object detection using YOLOv8
- Extracts detected regions with optional padding
- Supports batch processing
- Compatible with COCO dataset format

### CLIP Embeddings (`clip_embeddings.py`)
- Computes image and text embeddings
- Measures semantic similarity between regions and class labels
- Provides alignment scores between YOLO predictions and CLIP understanding
- Identifies potential misclassifications

### Grad-CAM (`gradcam_yolo.py`)
- Generates visual saliency maps for YOLOv8
- Highlights image regions that influenced predictions
- Supports multiple CAM variants (Grad-CAM, Grad-CAM++, Score-CAM)
- Provides region importance analysis

### LLaVA Explainer (`llava_explainer.py`)
- Generates natural language explanations
- Describes visual features supporting classification
- Compares conflicting predictions
- Includes fallback rule-based explainer

### Evaluation (`evaluation.py`)
- **Alignment Metrics**: Text-saliency overlap, spatial consistency, feature coverage
- **Faithfulness Metrics**: Occlusion sensitivity, counterfactual testing
- **Quality Metrics**: Specificity, conciseness, coherence, grounding

## Evaluation Metrics

### Alignment
Measures how well explanation text aligns with visual saliency:
- Text-saliency overlap score
- Spatial reference consistency
- Visual keyword precision/recall

### Faithfulness
Tests if explanations reflect model's actual reasoning:
- **Occlusion Test**: Masking salient regions should drop confidence
- **Counterfactual Test**: Masking non-salient regions should preserve confidence

### Quality
Evaluates explanation text quality:
- Specificity of visual descriptions
- Conciseness (30-60 words ideal)
- Logical coherence
- Grounding in visual evidence

## Output Format

### JSON Results
```json
{
  "image_path": "path/to/image.jpg",
  "num_detections": 3,
  "processing_time": 2.5,
  "detections": [
    {
      "bbox": [100, 100, 250, 250],
      "class_name": "dog",
      "confidence": 0.92,
      "semantic": {
        "alignment_score": 0.85,
        "top_clip_classes": [["dog", 0.91], ["cat", 0.72]]
      },
      "explanation": {
        "text": "This appears to be a dog based on its brown fur...",
        "visual_features": ["brown fur", "four legs", "tail"]
      },
      "evaluation": {
        "overall_score": 0.78,
        "alignment": {"mean_alignment": 0.82},
        "faithfulness": {"occlusion_sensitivity": 0.75}
      }
    }
  ]
}
```

### Visualizations
- Detection boxes with labels
- Grad-CAM heatmap overlays
- Cropped regions with explanations
- Evaluation metric plots

## COCO Dataset Setup

Download MS COCO for evaluation:
```bash
mkdir -p data/coco
cd data/coco

# Download validation images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

Expected structure:
```
data/coco/
├── val2017/
│   ├── 000000000139.jpg
│   └── ...
└── annotations/
    ├── instances_val2017.json
    └── ...
```

## Example Usage in Python

```python
from pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline(use_gpu=True, use_llava=False)

# Process image
result = pipeline.process_image(
    "test.jpg",
    max_detections=10,
    generate_explanations=True,
    compute_saliency=True,
    run_evaluation=True
)

# Access results
for det in result.detections:
    print(f"Class: {det.detection.class_name}")
    print(f"Confidence: {det.detection.confidence:.1%}")
    
    if det.explanation:
        print(f"Explanation: {det.explanation.explanation_text}")
    
    if det.evaluation:
        print(f"Quality: {det.evaluation.get_overall_score():.2f}")

# Save visualization
pipeline.visualize_result(result, "output.png")
```

## Performance Considerations

| Component | GPU Memory | Time per Image |
|-----------|------------|----------------|
| YOLOv8n   | ~1 GB      | ~50ms         |
| CLIP ViT-B/32 | ~1 GB  | ~30ms         |
| Grad-CAM  | ~500 MB    | ~100ms        |
| LLaVA 7B (4-bit) | ~6 GB | ~2s per detection |

**Tips:**
- Use `yolov8n.pt` for faster inference
- Disable LLaVA (`--no_llava`) for lightweight deployment
- Use `--no_saliency` to skip Grad-CAM if not needed
- Process in batches for large datasets

## Troubleshooting

### CUDA Out of Memory
- Use smaller models (`yolov8n.pt`, `ViT-B/32`)
- Disable LLaVA
- Reduce `--max_detections`
- Use `--cpu` mode

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### COCO Data Not Found
- Check directory structure matches expected format
- Ensure images are in `val2017/` subdirectory
- Verify annotations file exists

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{explainable_detection_2024,
  title={Explainable Object Detection Framework},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MS COCO Dataset](https://cocodataset.org/)

