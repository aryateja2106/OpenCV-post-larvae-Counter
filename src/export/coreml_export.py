"""CoreML export utilities for converting PyTorch shrimp larvae detector to iOS.

This module provides functions to:
1. Export the lightweight MobileNetV3-based detector to ONNX format.
2. Convert the ONNX model to CoreML (.mlpackage) format for iOS deployment.
3. Apply quantization to reduce model size for on-device inference.

Usage:
    python -m src.export.coreml_export \\
        --weights outputs_mobile/model_mobile.pth \\
        --output ios/ShrimpCounter/ShrimpCounter/Models/ShrimpDetector.mlpackage
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from src.logger import logging
from src.exception import CustomException
from src.components.lightweight_model import (
    LightweightDetector,
    LightweightModelConfig,
    load_lightweight_model,
    get_model_size_mb,
)


class CoreMLExportableWrapper(nn.Module):
    """Wrapper that makes the detection model exportable to CoreML.

    The standard Faster R-CNN model uses complex post-processing that
    is not directly exportable to CoreML/ONNX. This wrapper extracts
    the backbone + FPN + RPN + ROI heads into a simplified forward
    pass that produces bounding boxes, scores, and labels as flat
    tensors suitable for CoreML's NMS layer.
    """

    def __init__(self, model: nn.Module, score_threshold: float = 0.5):
        super().__init__()
        self.backbone = model.backbone
        self.rpn = model.rpn
        self.roi_heads = model.roi_heads
        self.transform = model.transform
        self.score_threshold = score_threshold

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing bounding box coordinates and confidence scores.

        Args:
            image: Input image tensor of shape (3, H, W), normalized to [0, 1].

        Returns:
            Tuple of (boxes, scores) where:
                boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2] format.
                scores: Tensor of shape (N,) with detection confidence scores.
        """
        image_list, _ = self.transform([image])
        features = self.backbone(image_list.tensors)
        proposals, _ = self.rpn(image_list, features)
        detections, _ = self.roi_heads(features, proposals, image_list.image_sizes)

        boxes = detections[0]["boxes"]
        scores = detections[0]["scores"]

        return boxes, scores


def export_to_onnx(
    weights_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (640, 640),
    config: Optional[LightweightModelConfig] = None,
) -> str:
    """Export the lightweight detector to ONNX format.

    Args:
        weights_path: Path to the trained model weights (.pth).
        output_path: Path for the output ONNX file.
        input_size: Input image dimensions (height, width).
        config: Model configuration.

    Returns:
        Path to the saved ONNX model.
    """
    try:
        detector = load_lightweight_model(weights_path, config)
        detector.model.eval()

        wrapper = CoreMLExportableWrapper(
            detector.model, detector.config.SCORE_THRESHOLD
        )
        wrapper.eval()

        dummy_input = torch.randn(3, input_size[0], input_size[1])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            opset_version=13,
            input_names=["image"],
            output_names=["boxes", "scores"],
            dynamic_axes={
                "boxes": {0: "num_detections"},
                "scores": {0: "num_detections"},
            },
        )

        logging.info(f"ONNX model exported to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error exporting to ONNX: {e}")
        raise CustomException(e, sys)


def export_to_coreml(
    weights_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (640, 640),
    quantize: bool = True,
    config: Optional[LightweightModelConfig] = None,
) -> str:
    """Export the lightweight detector to CoreML format for iOS deployment.

    Converts the PyTorch model through the ONNX intermediate format
    to a CoreML .mlpackage file. Optionally applies float16 quantization
    to reduce the model size by approximately 50%.

    Args:
        weights_path: Path to the trained model weights (.pth).
        output_path: Path for the output .mlpackage directory.
        input_size: Input image dimensions (height, width).
        quantize: Whether to apply float16 quantization.
        config: Model configuration.

    Returns:
        Path to the saved CoreML model.
    """
    try:
        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils

        onnx_path = output_path.replace(".mlpackage", ".onnx")
        export_to_onnx(weights_path, onnx_path, input_size, config)

        mlmodel = ct.converters.onnx.convert(model=onnx_path)

        mlmodel.author = "ShrimpSeedCounter"
        mlmodel.short_description = (
            "Lightweight shrimp larvae detector for real-time counting. "
            "Uses MobileNetV3-Large FPN backbone."
        )
        mlmodel.input_description["image"] = (
            "Input image (RGB, 640x640 pixels)"
        )
        mlmodel.output_description["boxes"] = (
            "Detected bounding boxes [x1, y1, x2, y2]"
        )
        mlmodel.output_description["scores"] = (
            "Detection confidence scores"
        )

        if quantize:
            mlmodel = quantization_utils.quantize_weights(
                mlmodel, nbits=16
            )
            logging.info("Applied float16 quantization")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        mlmodel.save(output_path)

        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        logging.info(f"CoreML model exported to {output_path}")
        return output_path

    except ImportError:
        logging.warning(
            "coremltools not installed. Install with: pip install coremltools"
        )
        logging.info("Falling back to ONNX export only")
        onnx_path = output_path.replace(".mlpackage", ".onnx")
        return export_to_onnx(weights_path, onnx_path, input_size, config)

    except Exception as e:
        logging.error(f"Error exporting to CoreML: {e}")
        raise CustomException(e, sys)


def export_torchscript(
    weights_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (640, 640),
    config: Optional[LightweightModelConfig] = None,
) -> str:
    """Export the model as TorchScript for cross-platform mobile deployment.

    TorchScript models can be used with PyTorch Mobile on both iOS and
    Android, providing an alternative to CoreML for cross-platform support.

    Args:
        weights_path: Path to the trained model weights (.pth).
        output_path: Path for the output .pt TorchScript file.
        input_size: Input image dimensions (height, width).
        config: Model configuration.

    Returns:
        Path to the saved TorchScript model.
    """
    try:
        detector = load_lightweight_model(weights_path, config)
        detector.model.eval()

        scripted_model = torch.jit.script(detector.model)

        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(
            scripted_model
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        optimized_model._save_for_lite_interpreter(output_path)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(
            f"TorchScript model exported to {output_path} "
            f"({file_size_mb:.2f} MB)"
        )
        return output_path

    except Exception as e:
        logging.error(f"Error exporting TorchScript: {e}")
        raise CustomException(e, sys)


def main():
    """Command-line entry point for model export."""
    parser = argparse.ArgumentParser(
        description="Export shrimp larvae detector to mobile formats"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ios/ShrimpCounter/ShrimpCounter/Models/ShrimpDetector.mlpackage",
        help="Output path for exported model",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["coreml", "onnx", "torchscript"],
        default="coreml",
        help="Export format",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input image size (height width)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable float16 quantization for CoreML export",
    )

    args = parser.parse_args()
    input_size = tuple(args.input_size)

    if args.format == "coreml":
        export_to_coreml(
            args.weights,
            args.output,
            input_size,
            quantize=not args.no_quantize,
        )
    elif args.format == "onnx":
        output = args.output.replace(".mlpackage", ".onnx")
        export_to_onnx(args.weights, output, input_size)
    elif args.format == "torchscript":
        output = args.output.replace(".mlpackage", ".ptl")
        export_torchscript(args.weights, output, input_size)


if __name__ == "__main__":
    main()
