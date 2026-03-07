"""Mobile-optimized prediction pipeline for shrimp larvae detection.

This pipeline uses the lightweight MobileNetV3-based detector instead of
Detectron2, enabling deployment on mobile devices and edge hardware.
It provides the same ROI masking, sliding window, and NMS functionality
as the original prediction pipeline but with pure torchvision inference.
"""

import cv2
import numpy as np
import torch
import supervision as sv
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.lightweight_model import (
    LightweightDetector,
    LightweightModelConfig,
    load_lightweight_model,
)


@dataclass
class ROIConfig:
    """Region of interest configuration."""
    shape: str
    points: List[Tuple[float, float]]
    active: bool = True


class MobilePredictionPipeline:
    """Prediction pipeline using the lightweight MobileNetV3 detector.

    This pipeline mirrors the original PredictionPipeline API but uses
    the lightweight torchvision-based model instead of Detectron2,
    making it suitable for mobile and edge deployment.
    """

    def __init__(self):
        self.detector = None
        self.device = None

    def initialize_model(
        self,
        weights_path: str,
        conf_threshold: float = 0.5,
        config: Optional[LightweightModelConfig] = None,
    ):
        """Initialize the lightweight detection model.

        Args:
            weights_path: Path to model weights (.pth file).
            conf_threshold: Minimum confidence score for detections.
            config: Model configuration overrides.
        """
        try:
            config = config or LightweightModelConfig()
            config.SCORE_THRESHOLD = conf_threshold

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.detector = load_lightweight_model(weights_path, config)
            self.detector.model.to(self.device)
            self.detector.model.eval()

            logging.info(
                f"Mobile model initialized on {self.device}"
            )

        except Exception as e:
            logging.error(f"Error initializing mobile model: {e}")
            raise CustomException(e, sys)

    def create_roi_mask(
        self, image_shape: Tuple[int, ...], roi_config: ROIConfig
    ) -> np.ndarray:
        """Create mask for region of interest.

        Args:
            image_shape: Shape of the input image (H, W, C).
            roi_config: ROI configuration with shape and points.

        Returns:
            Binary mask array of shape (H, W).
        """
        try:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            points = np.array(roi_config.points, dtype=np.int32)

            if roi_config.shape == "polygon":
                cv2.fillPoly(mask, [points], 255)
            elif roi_config.shape == "rectangle":
                cv2.rectangle(
                    mask, tuple(points[0]), tuple(points[1]), 255, -1
                )
            elif roi_config.shape == "circle":
                center = tuple(points[0])
                radius = int(
                    np.sqrt(
                        (points[1][0] - points[0][0]) ** 2
                        + (points[1][1] - points[0][1]) ** 2
                    )
                )
                cv2.circle(mask, center, radius, 255, -1)

            return mask

        except Exception as e:
            logging.error(f"Error creating ROI mask: {e}")
            raise CustomException(e, sys)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized tensor for inference.

        Args:
            image: Input BGR image as numpy array (H, W, 3).

        Returns:
            Normalized float32 tensor of shape (3, H, W) in [0, 1].
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        return tensor

    @torch.no_grad()
    def process_batch(
        self, image_batch: List[np.ndarray]
    ) -> List[sv.Detections]:
        """Process a batch of images through the detector.

        Args:
            image_batch: List of BGR images as numpy arrays.

        Returns:
            List of sv.Detections for each image.
        """
        try:
            if self.detector is None:
                raise ValueError("Model not initialized. Call initialize_model() first.")

            results = []
            tensors = [
                self._preprocess_image(img).to(self.device)
                for img in image_batch
            ]

            outputs = self.detector.model(tensors)

            for output in outputs:
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                if len(boxes) == 0:
                    results.append(sv.Detections.empty())
                    continue

                results.append(
                    sv.Detections(
                        xyxy=boxes,
                        confidence=scores,
                        class_id=labels,
                    )
                )

            return results

        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            raise CustomException(e, sys)

    def process_image(
        self,
        image: np.ndarray,
        roi_configs: List[ROIConfig],
        slice_size: int = 450,
        overlap_ratio: float = 0.65,
        iou_threshold: float = 0.4,
        max_area: float = 900.0,
    ) -> sv.Detections:
        """Process a single image with ROI masking and sliding window.

        Args:
            image: Input BGR image as numpy array.
            roi_configs: List of ROI configurations.
            slice_size: Size of each sliding window tile.
            overlap_ratio: Overlap between adjacent tiles.
            iou_threshold: IoU threshold for NMS merging.
            max_area: Maximum bounding box area to keep.

        Returns:
            Filtered and merged detections.
        """
        try:
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for roi_config in roi_configs:
                if roi_config.active:
                    mask = self.create_roi_mask(image.shape, roi_config)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            masked_image = image.copy()
            if np.any(combined_mask):
                masked_image[combined_mask == 0] = 0

            def slicer_callback(
                slice_img: np.ndarray,
            ) -> sv.Detections:
                results = self.process_batch([slice_img])[0]
                return _filter_large_boxes(results, max_area)

            slicer = sv.InferenceSlicer(
                callback=slicer_callback,
                slice_wh=(slice_size, slice_size),
                overlap_ratio_wh=(overlap_ratio, overlap_ratio),
            )

            detections = slicer(masked_image)
            return _merge_detections(detections, iou_threshold)

        except Exception as e:
            logging.error(f"Error processing image: {e}")
            raise CustomException(e, sys)


def _filter_large_boxes(
    detections: sv.Detections, max_area: float
) -> sv.Detections:
    """Filter out bounding boxes larger than max_area."""
    try:
        if len(detections) == 0:
            return sv.Detections.empty()

        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        for box, score, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
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
            class_id=np.array(filtered_classes),
        )

    except Exception as e:
        logging.error(f"Error filtering boxes: {e}")
        raise CustomException(e, sys)


def _merge_detections(
    detections: sv.Detections, iou_threshold: float
) -> sv.Detections:
    """Merge overlapping detections using NMS."""
    try:
        return sv.Detections.merge(detections, iou_threshold)
    except Exception as e:
        logging.error(f"Error merging detections: {e}")
        raise CustomException(e, sys)
