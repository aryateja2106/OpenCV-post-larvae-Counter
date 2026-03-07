import os
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from src.logger import logging
from src.exception import CustomException


@dataclass
class LightweightModelConfig:
    """Configuration for the lightweight MobileNet-based detection model.

    This model is designed for mobile/edge deployment, reducing the model
    size from ~496MB (Faster R-CNN R50-FPN) to ~72MB (~36MB quantized)
    while maintaining competitive accuracy for shrimp larvae detection.
    """
    NUM_CLASSES: int = 2  # background + shrimp_larva
    INPUT_SIZE: Tuple[int, int] = (640, 640)
    SCORE_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.4
    MAX_DETECTIONS: int = 500
    LEARNING_RATE: float = 0.005
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    NUM_EPOCHS: int = 50
    BATCH_SIZE: int = 4
    OUTPUT_DIR: str = "outputs_mobile"


class LightweightDetector(nn.Module):
    """Lightweight shrimp larvae detector using MobileNetV3-Large backbone.

    Uses torchvision's Faster R-CNN with MobileNetV3-Large FPN backbone,
    producing a model ~20x smaller than the ResNet-50 FPN variant.
    Suitable for CoreML export and on-device iOS inference.
    """

    def __init__(self, config: Optional[LightweightModelConfig] = None):
        super().__init__()
        try:
            self.config = config or LightweightModelConfig()
            self.model = self._build_model()
            logging.info("LightweightDetector initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing LightweightDetector: {e}")
            raise CustomException(e, sys)

    def _build_model(self) -> nn.Module:
        """Build the MobileNetV3-Large FPN Faster R-CNN model."""
        try:
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights="DEFAULT",
                trainable_backbone_layers=3
            )

            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.config.NUM_CLASSES
            )

            model.roi_heads.score_thresh = self.config.SCORE_THRESHOLD
            model.roi_heads.nms_thresh = self.config.NMS_THRESHOLD
            model.roi_heads.detections_per_img = self.config.MAX_DETECTIONS

            logging.info(
                f"Built MobileNetV3-Large FPN model with "
                f"{self.config.NUM_CLASSES} classes"
            )
            return model

        except Exception as e:
            logging.error(f"Error building model: {e}")
            raise CustomException(e, sys)

    def forward(self, images, targets=None):
        """Forward pass through the detection model."""
        return self.model(images, targets)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups for fine-tuning."""
        try:
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                params,
                lr=self.config.LEARNING_RATE,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY
            )
            logging.info("Optimizer created successfully")
            return optimizer

        except Exception as e:
            logging.error(f"Error creating optimizer: {e}")
            raise CustomException(e, sys)

    def get_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warm-up and step decay."""
        try:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[20, 35, 45],
                gamma=0.1
            )
            logging.info("LR scheduler created successfully")
            return scheduler

        except Exception as e:
            logging.error(f"Error creating LR scheduler: {e}")
            raise CustomException(e, sys)


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    buffer_size = sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )
    return (param_size + buffer_size) / (1024 * 1024)


def train_lightweight_model(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    config: Optional[LightweightModelConfig] = None
) -> LightweightDetector:
    """Train the lightweight detector on a shrimp larvae dataset.

    Args:
        train_dataset: Training dataset with images and COCO-format annotations.
        val_dataset: Optional validation dataset.
        config: Model and training configuration.

    Returns:
        Trained LightweightDetector instance.
    """
    try:
        config = config or LightweightModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on device: {device}")

        detector = LightweightDetector(config)
        detector.model.to(device)

        model_size = get_model_size_mb(detector.model)
        logging.info(f"Model size: {model_size:.2f} MB")

        optimizer = detector.get_optimizer()
        lr_scheduler = detector.get_lr_scheduler(optimizer)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        detector.model.train()
        for epoch in range(config.NUM_EPOCHS):
            epoch_loss = 0.0
            num_batches = 0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                loss_dict = detector.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                num_batches += 1

            lr_scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            logging.info(
                f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
                f"Loss: {avg_loss:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # Save model
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(config.OUTPUT_DIR, "model_mobile.pth")
        torch.save(detector.model.state_dict(), save_path)
        logging.info(f"Lightweight model saved to {save_path}")

        return detector

    except Exception as e:
        logging.error(f"Error training lightweight model: {e}")
        raise CustomException(e, sys)


def load_lightweight_model(
    weights_path: str,
    config: Optional[LightweightModelConfig] = None
) -> LightweightDetector:
    """Load a trained lightweight detector from saved weights.

    Args:
        weights_path: Path to saved model weights (.pth file).
        config: Model configuration.

    Returns:
        LightweightDetector with loaded weights.
    """
    try:
        config = config or LightweightModelConfig()
        detector = LightweightDetector(config)

        state_dict = torch.load(weights_path, map_location="cpu",
                                weights_only=True)
        detector.model.load_state_dict(state_dict)
        detector.model.eval()

        model_size = get_model_size_mb(detector.model)
        logging.info(
            f"Loaded lightweight model from {weights_path} "
            f"({model_size:.2f} MB)"
        )
        return detector

    except Exception as e:
        logging.error(f"Error loading lightweight model: {e}")
        raise CustomException(e, sys)
