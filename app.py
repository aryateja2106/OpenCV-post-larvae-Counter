import streamlit as st
import supervision as sv
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import time

st.set_page_config(page_title="Object Detection App (GPU)", layout="wide")

# Check GPU availability
def check_gpu():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        st.sidebar.success(f"Using GPU: {device}")
        return True
    else:
        st.sidebar.warning("GPU not available, using CPU")
        return False

def get_model(config_path: str, weights_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"
    
    predictor = DefaultPredictor(cfg)
    return predictor

def create_roi_mask(image_shape, polygon):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

@torch.cuda.amp.autocast()
def merge_overlapping_boxes(detections, iou_threshold=0.7):
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    merged_detections = []

    while detections:
        best_detection = detections.pop(0)
        merged_detections.append(best_detection)

        remaining_detections = []
        for detection in detections:
            iou = calculate_iou(best_detection[:4], detection[:4])
            if iou < iou_threshold:
                remaining_detections.append(detection)

        detections = remaining_detections
    
    return merged_detections

def filter_large_bounding_boxes(detections, max_area):
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    
    for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= max_area:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_classes.append(class_id)
    
    if not filtered_boxes:
        return sv.Detections.empty()
    
    detection_data = {
        "xyxy": np.array(filtered_boxes),
        "confidence": np.array(filtered_scores),
        "class_id": np.array(filtered_classes)
    }
    return sv.Detections(**detection_data)

@torch.cuda.amp.autocast()
def process_image(image, model, slice_size, overlap_ratio, iou_threshold, max_area):
    def slicer_callback(slice: np.ndarray) -> sv.Detections:
        if torch.cuda.is_available():
            slice_tensor = torch.from_numpy(slice).cuda()
        else:
            slice_tensor = torch.from_numpy(slice)
            
        outputs = model(slice)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.array([])
        scores = instances.scores.numpy() if instances.has("scores") else np.array([])
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.array([])

        if boxes.shape[0] == 0:
            return sv.Detections.empty()

        detections = []
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            class_id = classes[i] if i < len(classes) else -1
            detections.append([box[0], box[1], box[2], box[3], score, class_id])

        merged_detections = merge_overlapping_boxes(detections, iou_threshold)
        
        boxes = [d[:4] for d in merged_detections]
        scores = [d[4] for d in merged_detections]
        classes = [d[5] for d in merged_detections]
        
        detection_data = {
            "xyxy": np.array(boxes),
            "confidence": np.array(scores),
            "class_id": np.array(classes)
        }
        return sv.Detections(**detection_data)

    # Create slicer without the overlap_filter_strategy parameter
    slicer = sv.InferenceSlicer(
        callback=slicer_callback,
        slice_wh=(slice_size, slice_size),
        overlap_ratio_wh=(overlap_ratio, overlap_ratio)
    )
    
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            detections = slicer(image)
    else:
        detections = slicer(image)
        
    filtered_detections = filter_large_bounding_boxes(detections, max_area)
    
    return filtered_detections

def main():
    st.title("Object Detection with GPU Acceleration")
    
    is_gpu_available = check_gpu()
    
    if is_gpu_available:
        st.sidebar.info(f"CUDA Version: {torch.version.cuda}")
        st.sidebar.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    st.sidebar.header("Parameters")
    slice_size = st.sidebar.slider("Slice Size", 200, 800, 450)
    overlap_ratio = st.sidebar.slider("Overlap Ratio", 0.1, 0.9, 0.65)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.4)
    max_area = st.sidebar.slider("Max Bounding Box Area", 100, 2000, 900)
    
    if is_gpu_available:
        batch_size = st.sidebar.slider("Batch Size", 1, 8, 4)
    
    image_source = st.radio("Select Image Source", ["Upload Image", "Use Sample Image"])
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    else:
        sample_images = {
            "Sample 1": os.path.join('src','artifacts','test_samples','IMG_7254.JPG'),
            "Sample 2": os.path.join('src','artifacts','test_samples','IMG_7240.JPG'),
            "Sample 3": os.path.join('src','artifacts','test_samples','IMG_7235 (1).JPG')
        }
        selected_sample = st.selectbox("Choose a sample image", list(sample_images.keys()))
        if selected_sample:
            image_path = sample_images[selected_sample]
            uploaded_file = open(image_path, 'rb')
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Image Preview with ROI")
            if 'original_image' not in st.session_state:
                st.session_state.original_image = image.copy()
            
            def draw_roi_preview():
                try:
                    polygon = [tuple(map(int, point.strip().split(','))) 
                             for point in roi_text.strip().split('\n')]
                    
                    preview_image = st.session_state.original_image.copy()
                    
                    cv2.polylines(preview_image, 
                                [np.array(polygon, dtype=np.int32)],
                                True, (0, 255, 0), 2)
                    
                    overlay = preview_image.copy()
                    cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, preview_image, 0.7, 0, preview_image)
                    
                    return preview_image
                except Exception as e:
                    return st.session_state.original_image
            
        with col2:
            st.subheader("Region of Interest")
            st.write("Enter ROI coordinates (x,y) separated by commas, one point per line:")
            roi_text = st.text_area("ROI Coordinates", 
                                  "175,628\n515,330\n1305,330\n1624,640\n1648,1949\n1300,2270\n500,2260\n160,1940",
                                  key="roi_input")
            
            roi_color = st.color_picker("ROI Outline Color", "#00FF00")
            roi_opacity = st.slider("ROI Opacity", 0.0, 1.0, 0.3)
            
            if st.button("Validate ROI"):
                try:
                    polygon = [tuple(map(int, point.strip().split(','))) 
                             for point in roi_text.strip().split('\n')]
                    if len(polygon) < 3:
                        st.error("ROI must have at least 3 points")
                    else:
                        st.success(f"Valid ROI with {len(polygon)} points")
                except Exception as e:
                    st.error(f"Invalid ROI format: {str(e)}")
            
            st.write("Quick ROI Templates:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Rectangle"):
                    h, w = image.shape[:2]
                    margin = 50
                    template = f"{margin},{margin}\n{w-margin},{margin}\n{w-margin},{h-margin}\n{margin},{h-margin}"
                    st.session_state.roi_input = template
                    st.experimental_rerun()
            with col2:
                if st.button("Center Square"):
                    h, w = image.shape[:2]
                    size = min(h, w) // 3
                    cx, cy = w//2, h//2
                    template = f"{cx-size},{cy-size}\n{cx+size},{cy-size}\n{cx+size},{cy+size}\n{cx-size},{cy+size}"
                    st.session_state.roi_input = template
                    st.experimental_rerun()
            
            if st.checkbox("Show Coordinate Helper"):
                st.write("Click on the image to get coordinates:")
                st.markdown("""
                    <script>
                        const img = document.querySelector('img');
                        img.addEventListener('click', function(e) {
                            const rect = e.target.getBoundingClientRect();
                            const x = e.clientX - rect.left;
                            const y = e.clientY - rect.top;
                            alert(`Coordinates: ${Math.round(x)},${Math.round(y)}`);
                        });
                    </script>
                    """, unsafe_allow_html=True)
        
        preview_image = draw_roi_preview()
        st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB))
        
        if st.button("Process Image"):
            try:
                start_time = torch.cuda.Event(enable_timing=True) if is_gpu_available else time.time()
                end_time = torch.cuda.Event(enable_timing=True) if is_gpu_available else None
                
                if is_gpu_available:
                    start_time.record()
                
                polygon = [tuple(map(int, point.strip().split(','))) 
                          for point in roi_text.strip().split('\n')]
                
                mask = create_roi_mask(image.shape, polygon)
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                
                config_path = os.path.join("outputs","config.yaml")
                weights_path = os.path.join("outputs","model_final.pth")
                model = get_model(config_path, weights_path)
                
                with st.spinner('Processing image...'):
                    filtered_detections = process_image(
                        masked_image, model, slice_size, overlap_ratio, 
                        iou_threshold, max_area
                    )
                
                annotator = sv.BoundingBoxAnnotator()
                annotated_frame = annotator.annotate(
                    scene=image.copy(), 
                    detections=filtered_detections
                )
                
                if is_gpu_available:
                    end_time.record()
                    torch.cuda.synchronize()
                    processing_time = start_time.elapsed_time(end_time) / 1000
                else:
                    processing_time = time.time() - start_time
                
                st.subheader("Detection Results")
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                st.write(f"Number of detected objects: {len(filtered_detections.xyxy)}")
                st.write(f"Processing time: {processing_time:.2f} seconds")
                
                if is_gpu_available:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    st.write(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
                    st.write(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                
            finally:
                if is_gpu_available:
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
