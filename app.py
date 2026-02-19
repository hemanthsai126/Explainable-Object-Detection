"""
Explainable Object Detection - Streamlit Web Application
Interactive web interface for the complete pipeline
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
import json
import io
import base64
from pathlib import Path
import tempfile
import os

# Set page config first
st.set_page_config(
    page_title="Explainable Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .detection-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .explanation-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(use_llava: bool = False):
    """Load and cache the detection pipeline"""
    from config import get_default_config, update_config_for_cpu
    from pipeline import ExplainableDetectionPipeline
    
    config = get_default_config()
    
    # Check for GPU
    if not torch.cuda.is_available():
        config = update_config_for_cpu(config)
        st.sidebar.warning("⚠️ Running on CPU (slower)")
    else:
        st.sidebar.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    pipeline = ExplainableDetectionPipeline(config, use_llava=use_llava)
    return pipeline


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 for display"""
    img_pil = Image.fromarray(image)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def display_detection_card(det, idx: int):
    """Display a styled detection card"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if det.detection.region is not None:
                st.image(det.detection.region, caption=f"Region {idx+1}", use_container_width=True)
        
        with col2:
            st.markdown(f"### {det.detection.class_name}")
            st.metric("Confidence", f"{det.detection.confidence:.1%}")
            
            if det.semantic_analysis and det.semantic_analysis.semantic_match:
                match = det.semantic_analysis.semantic_match
                alignment_color = "🟢" if match.alignment_score > 0.6 else "🟡" if match.alignment_score > 0.4 else "🔴"
                st.write(f"{alignment_color} **CLIP Alignment:** {match.alignment_score:.3f}")
                
                if match.top_k_classes:
                    st.write("**🔍 CLIP Top Matches:**")
                    for cls, score in match.top_k_classes[:5]:
                        bar_color = "🔵" if cls == det.detection.class_name else "⚪"
                        st.caption(f"{bar_color} {cls}: {score:.3f}")
            
            if det.evaluation:
                score = det.evaluation.get_overall_score()
                st.metric("Quality Score", f"{score:.3f}")


def display_saliency_map(det, original_image: np.ndarray):
    """Display saliency map visualization"""
    if det.saliency_map:
        col1, col2 = st.columns(2)
        with col1:
            st.image(det.saliency_map.heatmap, caption="Heatmap", use_container_width=True, clamp=True)
        with col2:
            st.image(det.saliency_map.overlay, caption="Overlay", use_container_width=True)
    else:
        st.info("Saliency map not available")


def display_explanation(det):
    """Display explanation with styling"""
    if det.explanation:
        st.markdown(f"""
        <div class="explanation-box">
            <strong>🤖 AI Explanation:</strong><br>
            {det.explanation.explanation_text}
        </div>
        """, unsafe_allow_html=True)
        
        if det.explanation.visual_features:
            st.write("**Visual Features Identified:**")
            cols = st.columns(min(len(det.explanation.visual_features), 4))
            for i, feature in enumerate(det.explanation.visual_features[:4]):
                with cols[i]:
                    st.info(feature)
    else:
        st.info("No explanation generated")


def process_video_frame(frame: np.ndarray, pipeline, conf_threshold: float = 0.25) -> tuple:
    """Process a single video frame for detection"""
    # Run YOLO detection only (fast mode for video)
    detection_result = pipeline.detector.detect(frame, extract_regions=True)
    
    # Draw detections
    annotated = draw_detections_on_image(frame, detection_result.detections)
    
    return annotated, detection_result


def draw_detections_on_image(image: np.ndarray, detections) -> np.ndarray:
    """Draw detection boxes on image - handles both Detection and ExplainedDetection"""
    img = image.copy()
    
    # Color palette
    np.random.seed(42)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    for i, det in enumerate(detections):
        # Handle both Detection and ExplainedDetection objects
        if hasattr(det, 'detection'):
            # ExplainedDetection
            bbox = det.detection.bbox
            class_name = det.detection.class_name
            confidence = det.detection.confidence
        else:
            # Detection
            bbox = det.bbox
            class_name = det.class_name
            confidence = det.confidence
        
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Explainable Object Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Combining YOLOv8 + CLIP + Grad-CAM + LLaVA for Interpretable Detection</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    yolo_model = st.sidebar.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Larger models are more accurate but slower"
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    max_detections = st.sidebar.slider(
        "Max Detections",
        min_value=1,
        max_value=20,
        value=10,
        help="Maximum objects to analyze"
    )
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    compute_saliency = st.sidebar.checkbox("Compute Grad-CAM Saliency", value=True)
    generate_explanations = st.sidebar.checkbox("Generate Explanations", value=True)
    run_evaluation = st.sidebar.checkbox("Run Evaluation Metrics", value=True)
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        use_llava = st.checkbox(
            "Use LLaVA (requires >8GB VRAM)",
            value=False,
            help="Enable full LLaVA model for better explanations"
        )
    
    # Load pipeline with caching
    with st.spinner("Loading models..."):
        try:
            pipeline = load_pipeline(use_llava=use_llava)
            # Update config based on sidebar settings
            pipeline.config.yolo.model_size = yolo_model
            pipeline.config.yolo.confidence_threshold = conf_threshold
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📷 Image", "🎬 Video", "📹 Camera", "📊 Analysis", "📈 Evaluation", "ℹ️ About"
    ])
    
    with tab1:
        st.header("📷 Image Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Image upload
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Upload an image for object detection"
            )
            
            # Demo image option
            use_demo = st.checkbox("Use demo image instead")
            
            if use_demo:
                # Create demo image
                demo_img = np.zeros((480, 640, 3), dtype=np.uint8)
                demo_img[:] = (200, 200, 200)
                cv2.rectangle(demo_img, (50, 50), (180, 180), (255, 0, 0), -1)
                cv2.circle(demo_img, (350, 150), 80, (0, 255, 0), -1)
                cv2.rectangle(demo_img, (500, 100), (620, 250), (0, 0, 255), -1)
                cv2.ellipse(demo_img, (200, 350), (120, 60), 0, 0, 360, (255, 255, 0), -1)
                image = demo_img
                st.image(image, caption="Demo Image", use_container_width=True)
            elif uploaded_file is not None:
                # Load uploaded image
                image = np.array(Image.open(uploaded_file).convert("RGB"))
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                image = None
                st.info("👆 Upload an image or use the demo image to get started!")
        
        with col2:
            if image is not None:
                # Process button
                if st.button("🚀 Run Detection & Analysis", type="primary", use_container_width=True):
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Detection
                        status_text.text("🔍 Step 1/4: Running YOLOv8 detection...")
                        progress_bar.progress(10)
                        
                        start_time = time.time()
                        
                        # Run pipeline
                        status_text.text("🧠 Step 2/4: Computing CLIP embeddings...")
                        progress_bar.progress(30)
                        
                        status_text.text("🎨 Step 3/4: Generating Grad-CAM saliency...")
                        progress_bar.progress(50)
                        
                        status_text.text("📝 Step 4/4: Generating explanations...")
                        progress_bar.progress(70)
                        
                        result = pipeline.process_image(
                            image,
                            max_detections=max_detections,
                            generate_explanations=generate_explanations,
                            compute_saliency=compute_saliency,
                            run_evaluation=run_evaluation
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Store results in session state
                        st.session_state['result'] = result
                        st.session_state['original_image'] = image
                        st.session_state['processing_time'] = time.time() - start_time
                        
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"✅ Found {result.num_detections} objects in {st.session_state['processing_time']:.2f}s")
                        
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        progress_bar.empty()
                        status_text.empty()
                
                # Display results if available
                if 'result' in st.session_state:
                    result = st.session_state['result']
                    
                    # Annotated image
                    st.subheader("Detection Results")
                    annotated = draw_detections_on_image(result.image, result.detections)
                    st.image(annotated, caption=f"Detected {result.num_detections} objects", use_container_width=True)
    
    # ==================== VIDEO TAB ====================
    with tab2:
        st.header("🎬 Video Detection")
        st.markdown("Upload a video file for object detection on each frame.")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video for object detection"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            frame_skip = st.slider(
                "Process every N frames",
                min_value=1,
                max_value=30,
                value=5,
                help="Skip frames for faster processing"
            )
        
        with col2:
            max_frames = st.slider(
                "Max frames to process",
                min_value=10,
                max_value=500,
                value=100,
                help="Limit total frames processed"
            )
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            st.video(uploaded_video)
            
            if st.button("🚀 Process Video", type="primary", key="process_video"):
                # Open video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                st.info(f"Video: {total_frames} frames @ {fps} FPS")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_display = st.empty()
                
                # Results storage
                video_results = []
                processed_frames = []
                frame_count = 0
                processed_count = 0
                
                while cap.isOpened() and processed_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames
                    if frame_count % frame_skip != 0:
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    annotated_frame, detection_result = process_video_frame(
                        frame_rgb, pipeline, conf_threshold
                    )
                    
                    processed_frames.append(annotated_frame)
                    video_results.append({
                        'frame': frame_count,
                        'detections': len(detection_result.detections),
                        'classes': [d.class_name for d in detection_result.detections]
                    })
                    
                    processed_count += 1
                    
                    # Update progress
                    progress = min(processed_count / max_frames, frame_count / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({processed_count} processed)")
                    
                    # Show current frame
                    frame_display.image(annotated_frame, caption=f"Frame {frame_count}", use_container_width=True)
                
                cap.release()
                progress_bar.progress(1.0)
                status_text.text("✅ Video processing complete!")
                
                # Store results
                st.session_state['video_results'] = video_results
                st.session_state['video_frames'] = processed_frames
                
                # Summary
                st.success(f"Processed {processed_count} frames")
                
                total_detections = sum(r['detections'] for r in video_results)
                st.metric("Total Detections", total_detections)
                
                # Show frame browser
                if processed_frames:
                    st.subheader("Browse Processed Frames")
                    frame_idx = st.slider("Select Frame", 0, len(processed_frames)-1, 0)
                    st.image(processed_frames[frame_idx], 
                            caption=f"Frame {video_results[frame_idx]['frame']} - {video_results[frame_idx]['detections']} detections",
                            use_container_width=True)
                
                # Cleanup temp file
                os.unlink(video_path)
    
    # ==================== CAMERA TAB ====================
    with tab3:
        st.header("📹 Live Camera Detection")
        st.markdown("Use your webcam for real-time object detection.")
        
        # Camera input option
        camera_option = st.radio(
            "Select camera mode:",
            ["📸 Capture Single Frame", "🎥 Live Stream (Experimental)"],
            horizontal=True
        )
        
        if camera_option == "📸 Capture Single Frame":
            st.info("📷 Click the camera button below to capture a frame for detection.")
            
            camera_input = st.camera_input("Capture image from webcam")
            
            if camera_input is not None:
                # Convert to numpy array
                camera_image = np.array(Image.open(camera_input).convert("RGB"))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(camera_image, caption="Captured Image", use_container_width=True)
                
                with col2:
                    if st.button("🔍 Analyze Captured Image", type="primary"):
                        with st.spinner("Processing..."):
                            # Run detection
                            result = pipeline.process_image(
                                camera_image,
                                max_detections=max_detections,
                                generate_explanations=generate_explanations,
                                compute_saliency=compute_saliency,
                                run_evaluation=run_evaluation
                            )
                            
                            # Store results
                            st.session_state['result'] = result
                            st.session_state['original_image'] = camera_image
                            st.session_state['processing_time'] = result.processing_time
                            
                            # Show annotated result
                            annotated = draw_detections_on_image(result.image, result.detections)
                            st.image(annotated, caption=f"Detected {result.num_detections} objects", use_container_width=True)
                            
                            st.success(f"✅ Found {result.num_detections} objects! Check the Analysis tab for details.")
        
        else:  # Live Stream
            st.warning("⚠️ Live streaming requires continuous processing and may be resource-intensive.")
            
            run_stream = st.checkbox("🎥 Start Live Detection", value=False)
            
            if run_stream:
                st.info("Starting camera stream... Press 'Stop' or uncheck to end.")
                
                # Create placeholders
                frame_placeholder = st.empty()
                stats_placeholder = st.empty()
                stop_button = st.button("⏹️ Stop Stream")
                
                # Open camera
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open camera. Please check your webcam connection.")
                else:
                    frame_count = 0
                    fps_start = time.time()
                    
                    while run_stream and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from camera")
                            break
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process frame (detection only for speed)
                        annotated_frame, detection_result = process_video_frame(
                            frame_rgb, pipeline, conf_threshold
                        )
                        
                        frame_count += 1
                        
                        # Calculate FPS
                        elapsed = time.time() - fps_start
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        # Display frame
                        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                        
                        # Display stats
                        stats_placeholder.markdown(f"""
                        **Live Stats:** 
                        - Detections: {len(detection_result.detections)} 
                        - FPS: {current_fps:.1f} 
                        - Frames: {frame_count}
                        """)
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.03)
                    
                    cap.release()
                    st.info("Camera stream stopped.")
    
    # ==================== ANALYSIS TAB ====================
    with tab4:
        st.header("📊 Detailed Analysis")
        
        if 'result' not in st.session_state:
            st.info("👈 Upload and process an image first!")
        else:
            result = st.session_state['result']
            original_image = st.session_state['original_image']
            
            # Summary metrics
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Objects Detected", result.num_detections)
            with col2:
                st.metric("Processing Time", f"{st.session_state['processing_time']:.2f}s")
            with col3:
                if result.detections:
                    avg_conf = np.mean([d.detection.confidence for d in result.detections])
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
            with col4:
                if result.detections and result.detections[0].evaluation:
                    avg_score = np.mean([d.evaluation.get_overall_score() 
                                        for d in result.detections if d.evaluation])
                    st.metric("Avg Quality", f"{avg_score:.3f}")
            
            st.divider()
            
            # Individual detections
            st.subheader("Detection Details")
            
            for i, det in enumerate(result.detections):
                with st.expander(f"🎯 Detection {i+1}: {det.detection.class_name} ({det.detection.confidence:.1%})", expanded=(i==0)):
                    
                    tab_a, tab_b, tab_c = st.tabs(["📋 Overview", "🔥 Saliency", "💬 Explanation"])
                    
                    with tab_a:
                        display_detection_card(det, i)
                    
                    with tab_b:
                        display_saliency_map(det, original_image)
                    
                    with tab_c:
                        display_explanation(det)
    
    # ==================== EVALUATION TAB ====================
    with tab5:
        st.header("📈 Evaluation Metrics")
        
        if 'result' not in st.session_state:
            st.info("👈 Upload and process an image first!")
        else:
            result = st.session_state['result']
            
            # Check if evaluation was run
            evaluations = [d.evaluation for d in result.detections if d.evaluation]
            
            if not evaluations:
                st.warning("⚠️ Evaluation metrics not available. Enable 'Run Evaluation Metrics' in sidebar.")
            else:
                # Collect metrics
                alignment_scores = []
                faithfulness_scores = []
                quality_scores = []
                overall_scores = []
                
                for e in evaluations:
                    alignment_scores.append(e.alignment.to_dict()['mean_alignment'])
                    faithfulness_scores.append(e.faithfulness.to_dict()['mean_faithfulness'])
                    quality_scores.append(e.quality.to_dict()['mean_quality'])
                    overall_scores.append(e.get_overall_score())
                
                # Summary statistics
                st.subheader("Overall Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Alignment",
                        f"{np.mean(alignment_scores):.3f}",
                        help="How well explanations align with saliency maps"
                    )
                with col2:
                    st.metric(
                        "Faithfulness",
                        f"{np.mean(faithfulness_scores):.3f}",
                        help="How faithful explanations are to model reasoning"
                    )
                with col3:
                    st.metric(
                        "Quality",
                        f"{np.mean(quality_scores):.3f}",
                        help="Text quality of explanations"
                    )
                with col4:
                    st.metric(
                        "Overall",
                        f"{np.mean(overall_scores):.3f}",
                        help="Combined quality score"
                    )
                
                st.divider()
                
                # Charts
                st.subheader("Score Distribution")
                
                import plotly.graph_objects as go
                import plotly.express as px
                
                # Bar chart per detection
                fig = go.Figure()
                
                classes = [d.detection.class_name for d in result.detections if d.evaluation]
                
                fig.add_trace(go.Bar(name='Alignment', x=classes, y=alignment_scores, marker_color='#667eea'))
                fig.add_trace(go.Bar(name='Faithfulness', x=classes, y=faithfulness_scores, marker_color='#764ba2'))
                fig.add_trace(go.Bar(name='Quality', x=classes, y=quality_scores, marker_color='#f093fb'))
                fig.add_trace(go.Bar(name='Overall', x=classes, y=overall_scores, marker_color='#f5576c'))
                
                fig.update_layout(
                    barmode='group',
                    title='Evaluation Metrics by Detection',
                    xaxis_title='Detection',
                    yaxis_title='Score',
                    yaxis_range=[0, 1],
                    legend_title='Metric'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                st.subheader("Detailed Metrics")
                
                for i, det in enumerate(result.detections):
                    if det.evaluation:
                        with st.expander(f"📊 {det.detection.class_name} - Detailed Metrics"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Alignment Metrics**")
                                align_dict = det.evaluation.alignment.to_dict()
                                for key, value in align_dict.items():
                                    if isinstance(value, float):
                                        st.write(f"• {key}: {value:.3f}")
                            
                            with col2:
                                st.markdown("**Faithfulness Metrics**")
                                faith_dict = det.evaluation.faithfulness.to_dict()
                                for key, value in faith_dict.items():
                                    if isinstance(value, float):
                                        st.write(f"• {key}: {value:.3f}")
                            
                            with col3:
                                st.markdown("**Quality Metrics**")
                                qual_dict = det.evaluation.quality.to_dict()
                                for key, value in qual_dict.items():
                                    if isinstance(value, float):
                                        st.write(f"• {key}: {value:.3f}")
                
                # Export results
                st.divider()
                st.subheader("Export Results")
                
                if st.button("📥 Download Results as JSON"):
                    result_dict = result.to_dict()
                    json_str = json.dumps(result_dict, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="detection_results.json",
                        mime="application/json"
                    )
    
    # ==================== ABOUT TAB ====================
    with tab6:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🔍 Explainable Object Detection Framework
        
        This application combines state-of-the-art computer vision models with explainability 
        techniques to provide interpretable object detection results.
        
        #### 🛠️ Components
        
        | Component | Purpose |
        |-----------|---------|
        | **YOLOv8** | Fast and accurate object detection |
        | **CLIP** | Visual-text semantic grounding |
        | **Grad-CAM** | Visual saliency maps |
        | **LLaVA** | Natural language explanations |
        
        #### 📊 Evaluation Metrics
        
        - **Alignment**: Measures how well the explanation text aligns with visual saliency maps
        - **Faithfulness**: Tests if masking salient regions affects model confidence
        - **Quality**: Evaluates specificity, coherence, and grounding of explanations
        
        #### 🚀 How to Use
        
        1. Upload an image or use the demo image
        2. Configure detection settings in the sidebar
        3. Click "Run Detection & Analysis"
        4. Explore results in the Analysis and Evaluation tabs
        
        #### 📚 References
        
        - [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
        - [CLIP by OpenAI](https://github.com/openai/CLIP)
        - [LLaVA](https://github.com/haotian-liu/LLaVA)
        - [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
        """)
        
        st.divider()
        
        # System info
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**PyTorch Version:** {torch.__version__}")
            st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
                st.write(f"**CUDA Version:** {torch.version.cuda}")
        
        with col2:
            import sys
            st.write(f"**Python Version:** {sys.version.split()[0]}")
            st.write(f"**Streamlit Version:** {st.__version__}")


if __name__ == "__main__":
    main()

