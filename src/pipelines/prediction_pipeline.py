import cv2
import numpy as np
import torch
import supervision as sv
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from src.logger import logging
from src.exception import CustomException
import sys

@dataclass
class ROIConfig:
    shape: str
    points: List[Tuple[float, float]]
    active: bool = True

class PredictionPipeline:
    def __init__(self):
        self.predictor = None
        
    def initialize_model(self, config_path: str, weights_path: str, conf_threshold: float = 0.5):
        """Initialize the Detectron2 model"""
        try:
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
            
            if torch.cuda.is_available():
                cfg.MODEL.DEVICE = "cuda"
                logging.info("Using GPU for inference")
            else:
                cfg.MODEL.DEVICE = "cpu"
                logging.info("Using CPU for inference")
            
            self.predictor = DefaultPredictor(cfg)
            logging.info("Model initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise CustomException(e, sys)

    def create_roi_mask(self, image_shape: Tuple[int, int], roi_config: ROIConfig) -> np.ndarray:
        """Create mask for region of interest"""
        try:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            points = np.array(roi_config.points, dtype=np.int32)
            
            if roi_config.shape == "polygon":
                cv2.fillPoly(mask, [points], 255)
            elif roi_config.shape == "rectangle":
                cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), 255, -1)
            elif roi_config.shape == "circle":
                center = tuple(points[0])
                radius = int(np.sqrt(((points[1][0] - points[0][0]) ** 2) + 
                                   ((points[1][1] - points[0][1]) ** 2)))
                cv2.circle(mask, center, radius, 255, -1)
                
            return mask
            
        except Exception as e:
            logging.error(f"Error creating ROI mask: {str(e)}")
            raise CustomException(e, sys)

    @torch.cuda.amp.autocast()
    def process_batch(self, image_batch: List[np.ndarray]) -> List[sv.Detections]:
        """Process a batch of images"""
        try:
            if self.predictor is None:
                raise CustomException(e, sys)
            
            results = []
            for image in image_batch:
                outputs = self.predictor(image)
                instances = outputs["instances"].to("cpu")
                
                boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.array([])
                scores = instances.scores.numpy() if instances.has("scores") else np.array([])
                classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.array([])
                
                detection_data = {
                    "xyxy": boxes,
                    "confidence": scores,
                    "class_id": classes
                }
                results.append(sv.Detections(**detection_data))
                
            return results
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            raise CustomException(e, sys)

    def process_image(self, image: np.ndarray, roi_configs: List[ROIConfig], 
                     slice_size: int, overlap_ratio: float, iou_threshold: float, 
                     max_area: float) -> sv.Detections:
        """Process a single image with ROI and sliding window"""
        try:
            # Create combined ROI mask
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for roi_config in roi_configs:
                if roi_config.active:
                    mask = self.create_roi_mask(image.shape, roi_config)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply ROI mask
            masked_image = image.copy()
            if np.any(combined_mask):
                masked_image[combined_mask == 0] = 0
                
            # Create sliding window inference
            def slicer_callback(slice_img: np.ndarray) -> sv.Detections:
                results = self.process_batch([slice_img])[0]
                return filter_large_boxes(results, max_area)
            
            slicer = sv.InferenceSlicer(
                callback=slicer_callback,
                slice_wh=(slice_size, slice_size),
                overlap_ratio_wh=(overlap_ratio, overlap_ratio),
                batch_size=4
            )
            
            detections = slicer(masked_image)
            return merge_detections(detections, iou_threshold)
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise CustomException(e, sys)

def filter_large_boxes(detections: sv.Detections, max_area: float) -> sv.Detections:
    """Filter out bounding boxes larger than max_area"""
    try:
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []
        
        for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area <= max_area:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_classes.append(class_id)
        
        if not filtered_boxes:
            return sv.Detections.empty()
        
        return sv.Detections(
            xyxy=np.array(filtered_boxes),
            confidence=np.array(filtered_scores),
            class_id=np.array(filtered_classes)
        )
        
    except Exception as e:
        logging.error(f"Error filtering boxes: {str(e)}")
        raise CustomException(e, sys)

def merge_detections(detections: sv.Detections, iou_threshold: float) -> sv.Detections:
    """Merge overlapping detections using NMS"""
    try:
        return sv.Detections.merge(detections, iou_threshold)
    except Exception as e:
        logging.error(f"Error merging detections: {str(e)}")
        raise CustomException(e, sys)