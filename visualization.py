"""
Visualization Utilities for Explainable Object Detection
Creates comprehensive visual analysis outputs
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import seaborn as sns
from collections import defaultdict


def apply_colormap(heatmap: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """
    Apply colormap to heatmap
    
    Args:
        heatmap: Normalized heatmap [0, 1]
        colormap: Matplotlib colormap name
        
    Returns:
        Colored heatmap (RGB)
    """
    cm = plt.get_cmap(colormap)
    colored = cm(heatmap)[:, :, :3]  # Remove alpha
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.5, colormap: str = 'jet') -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Original image
        heatmap: Heatmap to overlay (will be resized)
        alpha: Blend factor (0 = image only, 1 = heatmap only)
        colormap: Colormap name
        
    Returns:
        Blended image
    """
    # Resize heatmap to match image
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Normalize if needed
    if heatmap_resized.max() > 1:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    
    # Apply colormap
    colored_heatmap = apply_colormap(heatmap_resized, colormap)
    
    # Blend
    blended = (alpha * colored_heatmap + (1 - alpha) * image).astype(np.uint8)
    
    return blended


def draw_detection_boxes(image: np.ndarray, detections, 
                        show_labels: bool = True,
                        show_confidence: bool = True,
                        line_thickness: int = 2,
                        font_scale: float = 0.6) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        detections: List of Detection or ExplainedDetection objects
        show_labels: Show class labels
        show_confidence: Show confidence scores
        line_thickness: Box line thickness
        font_scale: Font scale for labels
        
    Returns:
        Annotated image
    """
    img = image.copy()
    
    # Generate colors for classes
    np.random.seed(42)
    colors = {}
    
    for det in detections:
        # Handle both Detection and ExplainedDetection
        if hasattr(det, 'detection'):
            det = det.detection
        
        if det.class_name not in colors:
            colors[det.class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
        
        color = colors[det.class_name]
        x1, y1, x2, y2 = det.bbox
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if show_labels:
            label = det.class_name
            if show_confidence:
                label = f"{label}: {det.confidence:.0%}"
            
            # Background for label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def create_explanation_card(detection, width: int = 400, height: int = 300) -> np.ndarray:
    """
    Create a visual card showing explanation for a detection
    
    Args:
        detection: ExplainedDetection object
        width: Card width
        height: Card height
        
    Returns:
        Card image
    """
    # Create white background
    card = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    y_offset = 30
    
    # Title
    title = f"Detection: {detection.detection.class_name}"
    cv2.putText(card, title, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 2, cv2.LINE_AA)
    y_offset += 35
    
    # Confidence
    conf_text = f"Confidence: {detection.detection.confidence:.1%}"
    cv2.putText(card, conf_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (100, 100, 100), 1, cv2.LINE_AA)
    y_offset += 30
    
    # CLIP alignment
    if detection.semantic_analysis and detection.semantic_analysis.semantic_match:
        align = detection.semantic_analysis.semantic_match.alignment_score
        align_text = f"CLIP Alignment: {align:.2f}"
        color = (0, 150, 0) if align > 0.6 else (200, 100, 0) if align > 0.4 else (200, 0, 0)
        cv2.putText(card, align_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1, cv2.LINE_AA)
        y_offset += 30
    
    # Explanation
    if detection.explanation:
        cv2.putText(card, "Explanation:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 150), 1, cv2.LINE_AA)
        y_offset += 25
        
        # Wrap text
        text = detection.explanation.explanation_text
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            if tw < width - 20:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for line in lines[:5]:  # Max 5 lines
            cv2.putText(card, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (50, 50, 50), 1, cv2.LINE_AA)
            y_offset += 20
    
    # Quality score
    if detection.evaluation:
        score = detection.evaluation.get_overall_score()
        y_offset = height - 30
        score_text = f"Quality Score: {score:.2f}"
        color = (0, 150, 0) if score > 0.6 else (200, 100, 0) if score > 0.4 else (200, 0, 0)
        cv2.putText(card, score_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 2, cv2.LINE_AA)
    
    return card


def create_comprehensive_visualization(result, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create comprehensive visualization of pipeline results
    
    Args:
        result: PipelineResult object
        save_path: Optional path to save figure
        
    Returns:
        Visualization as numpy array
    """
    n_dets = min(len(result.detections), 6)
    
    if n_dets == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(result.image)
        ax.set_title("No Detections Found", fontsize=14)
        ax.axis('off')
    else:
        fig = plt.figure(figsize=(20, 5 * ((n_dets + 1) // 2)))
        gs = GridSpec(((n_dets + 1) // 2) + 1, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        # Original image with all detections
        ax_main = fig.add_subplot(gs[0, :2])
        img_with_boxes = draw_detection_boxes(result.image, result.detections)
        ax_main.imshow(img_with_boxes)
        ax_main.set_title(f"Original Image with Detections ({n_dets} objects)", fontsize=12)
        ax_main.axis('off')
        
        # Summary statistics
        ax_stats = fig.add_subplot(gs[0, 2:])
        ax_stats.axis('off')
        
        stats_text = f"Image Path: {result.image_path or 'N/A'}\n"
        stats_text += f"Processing Time: {result.processing_time:.2f}s\n"
        stats_text += f"Total Detections: {result.num_detections}\n\n"
        stats_text += "Class Distribution:\n"
        
        class_counts = defaultdict(int)
        for det in result.detections:
            class_counts[det.detection.class_name] += 1
        
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
            stats_text += f"  {cls}: {count}\n"
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Individual detection visualizations
        for i, det in enumerate(result.detections[:n_dets]):
            row = (i // 2) + 1
            col_start = (i % 2) * 2
            
            # Detection region + saliency
            ax_det = fig.add_subplot(gs[row, col_start])
            
            if det.saliency_map:
                ax_det.imshow(det.saliency_map.overlay)
            elif det.detection.region is not None:
                ax_det.imshow(det.detection.region)
            else:
                # Show cropped from original
                x1, y1, x2, y2 = det.detection.bbox
                cropped = result.image[y1:y2, x1:x2]
                ax_det.imshow(cropped)
            
            title = f"{det.detection.class_name} ({det.detection.confidence:.0%})"
            if det.semantic_analysis and det.semantic_analysis.semantic_match:
                align = det.semantic_analysis.semantic_match.alignment_score
                title += f"\nCLIP: {align:.2f}"
            ax_det.set_title(title, fontsize=10)
            ax_det.axis('off')
            
            # Explanation text
            ax_exp = fig.add_subplot(gs[row, col_start + 1])
            ax_exp.axis('off')
            
            if det.explanation:
                text = det.explanation.explanation_text
                wrapped = '\n'.join([text[j:j+45] for j in range(0, min(len(text), 200), 45)])
                
                ax_exp.text(0.05, 0.95, wrapped, transform=ax_exp.transAxes,
                           fontsize=9, verticalalignment='top', wrap=True,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                
                if det.evaluation:
                    score = det.evaluation.get_overall_score()
                    color = 'green' if score > 0.6 else 'orange' if score > 0.4 else 'red'
                    ax_exp.text(0.05, 0.05, f"Score: {score:.2f}", 
                               transform=ax_exp.transAxes, fontsize=10, color=color)
            else:
                ax_exp.text(0.5, 0.5, "No explanation", ha='center', va='center',
                           transform=ax_exp.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {save_path}")
    
    # Convert to array
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    return vis


def create_evaluation_plots(results: List, save_path: Optional[str] = None):
    """
    Create evaluation metric plots
    
    Args:
        results: List of PipelineResult objects
        save_path: Optional path to save figure
    """
    # Collect metrics
    alignment_scores = []
    faithfulness_scores = []
    quality_scores = []
    overall_scores = []
    class_scores = defaultdict(list)
    
    for result in results:
        for det in result.detections:
            if det.evaluation:
                alignment_scores.append(det.evaluation.alignment.to_dict()['mean_alignment'])
                faithfulness_scores.append(det.evaluation.faithfulness.to_dict()['mean_faithfulness'])
                quality_scores.append(det.evaluation.quality.to_dict()['mean_quality'])
                overall_scores.append(det.evaluation.get_overall_score())
                class_scores[det.detection.class_name].append(det.evaluation.get_overall_score())
    
    if not overall_scores:
        print("No evaluation data to plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Score distributions
    ax1 = fig.add_subplot(2, 3, 1)
    data = [alignment_scores, faithfulness_scores, quality_scores, overall_scores]
    labels = ['Alignment', 'Faithfulness', 'Quality', 'Overall']
    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title('Score Distributions')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of overall scores
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(overall_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(np.mean(overall_scores), color='red', linestyle='--', label=f'Mean: {np.mean(overall_scores):.2f}')
    ax2.set_title('Overall Score Distribution')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-class scores
    ax3 = fig.add_subplot(2, 3, 3)
    if class_scores:
        classes = list(class_scores.keys())[:10]  # Top 10 classes
        means = [np.mean(class_scores[c]) for c in classes]
        stds = [np.std(class_scores[c]) for c in classes]
        
        y_pos = np.arange(len(classes))
        ax3.barh(y_pos, means, xerr=stds, alpha=0.7, color='teal')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(classes)
        ax3.set_xlabel('Score')
        ax3.set_title('Scores by Class')
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Alignment vs Faithfulness scatter
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(alignment_scores, faithfulness_scores, alpha=0.5, c=overall_scores, 
               cmap='RdYlGn', s=50)
    ax4.set_xlabel('Alignment Score')
    ax4.set_ylabel('Faithfulness Score')
    ax4.set_title('Alignment vs Faithfulness')
    plt.colorbar(ax4.collections[0], ax=ax4, label='Overall')
    ax4.grid(True, alpha=0.3)
    
    # 5. Correlation heatmap
    ax5 = fig.add_subplot(2, 3, 5)
    corr_data = np.array([alignment_scores, faithfulness_scores, quality_scores, overall_scores])
    corr_matrix = np.corrcoef(corr_data)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               xticklabels=['Align', 'Faith', 'Quality', 'Overall'],
               yticklabels=['Align', 'Faith', 'Quality', 'Overall'],
               ax=ax5, vmin=-1, vmax=1)
    ax5.set_title('Metric Correlations')
    
    # 6. Summary statistics text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
    EVALUATION SUMMARY
    ══════════════════════════════
    
    Total Evaluations: {len(overall_scores)}
    
    Overall Scores:
      Mean:   {np.mean(overall_scores):.3f}
      Std:    {np.std(overall_scores):.3f}
      Min:    {np.min(overall_scores):.3f}
      Max:    {np.max(overall_scores):.3f}
    
    Success Rate (>0.6): {sum(1 for s in overall_scores if s > 0.6) / len(overall_scores):.1%}
    Failure Rate (<0.4): {sum(1 for s in overall_scores if s < 0.4) / len(overall_scores):.1%}
    
    Top Classes by Score:
    """
    
    top_classes = sorted([(c, np.mean(s)) for c, s in class_scores.items()],
                        key=lambda x: -x[1])[:5]
    for cls, score in top_classes:
        summary += f"\n      {cls}: {score:.3f}"
    
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Explainable Object Detection Evaluation Report', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved evaluation plots to {save_path}")
    
    plt.show()


def create_comparison_grid(images: List[np.ndarray], 
                          titles: List[str],
                          save_path: Optional[str] = None,
                          cols: int = 3) -> np.ndarray:
    """
    Create a grid of images for comparison
    
    Args:
        images: List of images
        titles: List of titles
        save_path: Optional save path
        cols: Number of columns
        
    Returns:
        Grid image
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to array
    fig.canvas.draw()
    grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    return grid


def create_saliency_comparison(image: np.ndarray, 
                              saliency_maps: List,
                              save_path: Optional[str] = None) -> np.ndarray:
    """
    Compare saliency maps for different detections
    
    Args:
        image: Original image
        saliency_maps: List of SaliencyMap objects
        save_path: Optional save path
        
    Returns:
        Comparison visualization
    """
    n = len(saliency_maps)
    
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('With All Boxes')
    axes[1, 0].axis('off')
    
    # Draw all boxes
    for sm in saliency_maps:
        x1, y1, x2, y2 = sm.bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 fill=False, edgecolor='red', linewidth=2)
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x1, y1-5, sm.class_name, color='red', fontsize=8)
    
    # Individual saliency maps
    for i, sm in enumerate(saliency_maps):
        # Heatmap
        axes[0, i+1].imshow(sm.heatmap, cmap='jet')
        axes[0, i+1].set_title(f'{sm.class_name}\n(conf: {sm.confidence:.0%})')
        axes[0, i+1].axis('off')
        
        # Overlay
        axes[1, i+1].imshow(sm.overlay)
        axes[1, i+1].set_title('Grad-CAM Overlay')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    return vis


def create_failure_analysis_visualization(failures: List[Dict],
                                         save_path: Optional[str] = None):
    """
    Visualize failure cases for analysis
    
    Args:
        failures: List of failure case dictionaries
        save_path: Optional save path
    """
    if not failures:
        print("No failures to visualize")
        return
    
    n = min(len(failures), 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, failure in enumerate(failures[:n]):
        ax = axes[i]
        ax.axis('off')
        
        text = f"Detection: {failure.get('class_name', 'N/A')}\n"
        text += f"Confidence: {failure.get('confidence', 0):.0%}\n"
        text += f"Overall Score: {failure.get('score', 0):.2f}\n"
        text += f"\nFailure Reasons:\n"
        
        for reason in failure.get('reasons', ['Unknown'])[:3]:
            text += f"  • {reason}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        
        ax.set_title(f"Failure Case {i+1}", fontsize=11)
    
    # Hide empty
    for i in range(n, 6):
        axes[i].axis('off')
    
    plt.suptitle('Failure Case Analysis', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved failure analysis to {save_path}")
    
    plt.show()


def save_result_gallery(results: List, output_dir: str):
    """
    Save gallery of all results
    
    Args:
        results: List of PipelineResult objects
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        # Save visualization
        vis_path = os.path.join(output_dir, f"result_{i:04d}.png")
        create_comprehensive_visualization(result, vis_path)
        
        # Save JSON
        json_path = os.path.join(output_dir, f"result_{i:04d}.json")
        result.save(json_path)
    
    print(f"Saved {len(results)} results to {output_dir}")


def run_demo():
    """Run visualization demo"""
    print("Visualization Demo")
    print("=" * 50)
    
    # Create sample data
    test_img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # Add some visual elements
    import cv2
    cv2.rectangle(test_img, (100, 100), (250, 250), (255, 100, 100), -1)
    cv2.circle(test_img, (450, 200), 80, (100, 255, 100), -1)
    
    # Test colormap application
    heatmap = np.random.rand(480, 640)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    colored = apply_colormap(heatmap)
    overlay = overlay_heatmap(test_img, heatmap)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(colored)
    axes[1].set_title('Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/viz_demo.png', dpi=150)
    print("Saved demo visualization to outputs/viz_demo.png")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    run_demo()

