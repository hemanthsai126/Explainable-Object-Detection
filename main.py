#!/usr/bin/env python3
"""
Explainable Object Detection Framework
Main entry point for running the complete pipeline

This framework integrates:
- YOLOv8 for object detection
- CLIP for visual-text semantic grounding
- LLaVA for natural language explanations
- Grad-CAM for visual saliency visualization

Usage:
    python main.py --image path/to/image.jpg
    python main.py --image_dir path/to/images/
    python main.py --demo
    python main.py --evaluate --data_dir path/to/coco/
"""

import argparse
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Explainable Object Detection Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run demo
    python main.py --demo
    
    # Process single image
    python main.py --image path/to/image.jpg --output outputs/
    
    # Process directory of images
    python main.py --image_dir path/to/images/ --output outputs/
    
    # Run evaluation on COCO
    python main.py --evaluate --data_dir data/coco/ --num_samples 100
    
    # Use LLaVA for explanations (requires more GPU memory)
    python main.py --image image.jpg --use_llava
    
    # CPU-only mode
    python main.py --image image.jpg --cpu
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--image', type=str, help='Path to single image')
    input_group.add_argument('--image_dir', type=str, help='Directory of images')
    input_group.add_argument('--demo', action='store_true', help='Run demo with test image')
    input_group.add_argument('--evaluate', action='store_true', help='Run evaluation on dataset')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='outputs/',
                       help='Output directory (default: outputs/)')
    parser.add_argument('--save_json', action='store_true', default=True,
                       help='Save results as JSON')
    parser.add_argument('--save_vis', action='store_true', default=True,
                       help='Save visualizations')
    
    # Model options
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 
                               'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model size (default: yolov8n.pt)')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       help='CLIP model (default: ViT-B/32)')
    parser.add_argument('--use_llava', action='store_true',
                       help='Use LLaVA for explanations (requires GPU)')
    
    # Processing options
    parser.add_argument('--max_detections', type=int, default=10,
                       help='Maximum detections per image (default: 10)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--no_saliency', action='store_true',
                       help='Skip Grad-CAM computation')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation metrics')
    
    # Dataset options (for evaluation)
    parser.add_argument('--data_dir', type=str, default='data/coco/',
                       help='COCO dataset directory')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for evaluation')
    
    # Hardware options
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    
    # Misc
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration from arguments"""
    from config import Config, YOLOConfig, CLIPConfig, LLaVAConfig, GradCAMConfig, EvaluationConfig, DataConfig
    
    # Determine device
    if args.cpu:
        device = 'cpu'
    elif args.device != 'auto':
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    config = Config(
        yolo=YOLOConfig(
            model_size=args.yolo_model,
            confidence_threshold=args.conf_threshold,
            device=device
        ),
        clip=CLIPConfig(
            model_name=args.clip_model,
            device=device
        ),
        llava=LLaVAConfig(
            device=device,
            load_in_4bit=device == 'cuda'
        ),
        gradcam=GradCAMConfig(
            use_cuda=device == 'cuda'
        ),
        evaluation=EvaluationConfig(
            num_samples_for_eval=args.num_samples,
            save_visualizations=args.save_vis
        ),
        data=DataConfig(
            coco_root=args.data_dir,
            output_dir=args.output
        )
    )
    
    return config, device


def run_demo(config, device, output_dir: str, use_llava: bool = False):
    """Run demonstration with synthetic test image"""
    print("\n" + "=" * 70)
    print("EXPLAINABLE OBJECT DETECTION DEMO")
    print("=" * 70)
    
    from pipeline import ExplainableDetectionPipeline
    from visualization import create_comprehensive_visualization
    import cv2
    
    # Initialize pipeline
    print(f"\nInitializing pipeline on {device}...")
    pipeline = ExplainableDetectionPipeline(config, use_llava=use_llava)
    
    # Create test image
    print("\nCreating synthetic test image...")
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    test_img[:] = (200, 200, 200)  # Light gray background
    
    # Add colored shapes
    cv2.rectangle(test_img, (50, 50), (200, 200), (0, 0, 255), -1)  # Blue square
    cv2.circle(test_img, (400, 150), 100, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(test_img, (550, 100), (750, 300), (255, 0, 0), -1)  # Red rectangle
    cv2.ellipse(test_img, (200, 450), (150, 80), 0, 0, 360, (255, 255, 0), -1)  # Yellow ellipse
    cv2.rectangle(test_img, (450, 350), (700, 550), (128, 0, 128), -1)  # Purple rectangle
    
    # Process
    print("\nProcessing image through pipeline...")
    result = pipeline.process_image(
        test_img,
        max_detections=10,
        generate_explanations=True,
        compute_saliency=not config.gradcam.use_cuda is False,
        run_evaluation=True
    )
    
    # Results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    print(f"Detections Found: {result.num_detections}")
    
    for i, det in enumerate(result.detections):
        print(f"\n[Detection {i+1}] {det.detection.class_name}")
        print(f"  Confidence: {det.detection.confidence:.1%}")
        print(f"  Bounding Box: {det.detection.bbox}")
        
        if det.semantic_analysis and det.semantic_analysis.semantic_match:
            match = det.semantic_analysis.semantic_match
            print(f"  CLIP Alignment: {match.alignment_score:.3f}")
            if match.top_k_classes:
                print(f"  CLIP Top Class: {match.top_k_classes[0][0]} ({match.top_k_classes[0][1]:.3f})")
        
        if det.explanation:
            exp_text = det.explanation.explanation_text
            print(f"  Explanation: {exp_text[:100]}{'...' if len(exp_text) > 100 else ''}")
        
        if det.evaluation:
            print(f"  Quality Score: {det.evaluation.get_overall_score():.3f}")
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualization
    vis_path = os.path.join(output_dir, "demo_visualization.png")
    create_comprehensive_visualization(result, vis_path)
    
    # JSON
    json_path = os.path.join(output_dir, "demo_results.json")
    result.save(json_path)
    
    print("\n" + "-" * 70)
    print(f"Results saved to {output_dir}")
    print("  - demo_visualization.png")
    print("  - demo_results.json")
    print("=" * 70 + "\n")
    
    return result


def process_single_image(config, device, image_path: str, output_dir: str, 
                        args, use_llava: bool = False):
    """Process a single image"""
    print(f"\nProcessing: {image_path}")
    
    from pipeline import ExplainableDetectionPipeline
    from visualization import create_comprehensive_visualization
    
    # Initialize
    pipeline = ExplainableDetectionPipeline(config, use_llava=use_llava)
    
    # Process
    result = pipeline.process_image(
        image_path,
        max_detections=args.max_detections,
        generate_explanations=True,
        compute_saliency=not args.no_saliency,
        run_evaluation=not args.no_eval
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    
    if args.save_vis:
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        create_comprehensive_visualization(result, vis_path)
    
    if args.save_json:
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        result.save(json_path)
    
    # Print summary
    print(f"  Detections: {result.num_detections}")
    print(f"  Time: {result.processing_time:.2f}s")
    
    return result


def process_directory(config, device, image_dir: str, output_dir: str,
                     args, use_llava: bool = False):
    """Process directory of images"""
    from pipeline import ExplainableDetectionPipeline
    from visualization import create_comprehensive_visualization, create_evaluation_plots
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"\nFound {len(image_paths)} images")
    
    # Initialize pipeline
    pipeline = ExplainableDetectionPipeline(config, use_llava=use_llava)
    
    # Process
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            result = pipeline.process_image(
                str(img_path),
                max_detections=args.max_detections,
                generate_explanations=True,
                compute_saliency=not args.no_saliency,
                run_evaluation=not args.no_eval
            )
            results.append(result)
            
            # Save individual results
            base_name = img_path.stem
            
            if args.save_vis:
                vis_path = os.path.join(output_dir, f"{base_name}_vis.png")
                create_comprehensive_visualization(result, vis_path)
            
            if args.save_json:
                json_path = os.path.join(output_dir, f"{base_name}.json")
                result.save(json_path)
                
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    # Generate evaluation plots if we have results
    if results and not args.no_eval:
        eval_plot_path = os.path.join(output_dir, "evaluation_summary.png")
        create_evaluation_plots(results, eval_plot_path)
        
        # Generate report
        report = pipeline.generate_report(results, 
                                         os.path.join(output_dir, "evaluation_report.json"))
        
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {len(results)}")
        print(f"Total detections: {sum(r.num_detections for r in results)}")
        
        if report:
            print(f"\nOverall Score: {report.get('overall', {}).get('mean', 0):.3f}")
            print(f"Alignment Score: {report.get('alignment', {}).get('mean', 0):.3f}")
            print(f"Faithfulness Score: {report.get('faithfulness', {}).get('mean', 0):.3f}")
            print(f"Quality Score: {report.get('quality', {}).get('mean', 0):.3f}")
    
    return results


def run_evaluation(config, device, args, use_llava: bool = False):
    """Run evaluation on COCO dataset"""
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION ON COCO DATASET")
    print("=" * 70)
    
    from pipeline import ExplainableDetectionPipeline
    from yolo_detector import COCODataLoader
    from visualization import create_evaluation_plots
    
    # Initialize
    print(f"\nInitializing pipeline on {device}...")
    pipeline = ExplainableDetectionPipeline(config, use_llava=use_llava)
    
    # Load data
    print(f"\nLoading COCO data from {args.data_dir}...")
    try:
        data_loader = COCODataLoader(args.data_dir, split="val2017")
        if len(data_loader) == 0:
            print("No images found. Please download COCO dataset.")
            print("Expected structure:")
            print(f"  {args.data_dir}/val2017/*.jpg")
            print(f"  {args.data_dir}/annotations/instances_val2017.json")
            return
    except Exception as e:
        print(f"Error loading COCO: {e}")
        print("\nTo download COCO dataset:")
        print("  wget http://images.cocodataset.org/zips/val2017.zip")
        print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        return
    
    # Sample images
    num_samples = min(args.num_samples, len(data_loader))
    sample_indices = np.random.choice(len(data_loader.image_ids), num_samples, replace=False)
    
    print(f"\nEvaluating on {num_samples} images...")
    
    results = []
    output_dir = os.path.join(args.output, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in tqdm(sample_indices, desc="Evaluating"):
        try:
            img_id = data_loader.image_ids[idx]
            img_path = data_loader.get_image_path(img_id)
            
            result = pipeline.process_image(
                str(img_path),
                max_detections=args.max_detections,
                generate_explanations=True,
                compute_saliency=not args.no_saliency,
                run_evaluation=True
            )
            results.append(result)
            
        except Exception as e:
            if args.verbose:
                print(f"\nError: {e}")
            continue
    
    # Generate report
    print("\nGenerating evaluation report...")
    
    report = pipeline.generate_report(results, 
                                     os.path.join(output_dir, "evaluation_report.json"))
    
    # Evaluation plots
    create_evaluation_plots(results, os.path.join(output_dir, "evaluation_plots.png"))
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Images Processed: {len(results)}")
    print(f"Total Detections: {sum(r.num_detections for r in results)}")
    
    if report:
        print(f"\nMetrics (mean ± std):")
        print(f"  Overall:     {report.get('overall', {}).get('mean', 0):.3f} ± "
              f"{report.get('overall', {}).get('std', 0):.3f}")
        print(f"  Alignment:   {report.get('alignment', {}).get('mean', 0):.3f} ± "
              f"{report.get('alignment', {}).get('std', 0):.3f}")
        print(f"  Faithfulness:{report.get('faithfulness', {}).get('mean', 0):.3f} ± "
              f"{report.get('faithfulness', {}).get('std', 0):.3f}")
        print(f"  Quality:     {report.get('quality', {}).get('mean', 0):.3f} ± "
              f"{report.get('quality', {}).get('std', 0):.3f}")
        
        print(f"\nTop Classes by Score:")
        by_class = report.get('by_class', {})
        for cls, score in sorted(by_class.items(), key=lambda x: -x[1])[:5]:
            print(f"  {cls}: {score:.3f}")
    
    print(f"\nResults saved to {output_dir}")
    print("=" * 70 + "\n")
    
    return results


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup
    config, device = setup_config(args)
    
    print("\n" + "=" * 70)
    print("EXPLAINABLE OBJECT DETECTION FRAMEWORK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"CLIP Model: {args.clip_model}")
    print(f"LLaVA: {'Enabled' if args.use_llava else 'Disabled'}")
    
    # Run appropriate mode
    if args.demo:
        run_demo(config, device, args.output, args.use_llava)
        
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        process_single_image(config, device, args.image, args.output, args, args.use_llava)
        
    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            sys.exit(1)
        process_directory(config, device, args.image_dir, args.output, args, args.use_llava)
        
    elif args.evaluate:
        run_evaluation(config, device, args, args.use_llava)
        
    else:
        print("\nNo input specified. Running demo...")
        print("Use --help for usage information.\n")
        run_demo(config, device, args.output, args.use_llava)


if __name__ == "__main__":
    main()

