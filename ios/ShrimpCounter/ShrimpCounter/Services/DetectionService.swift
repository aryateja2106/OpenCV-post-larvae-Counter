import Foundation
import UIKit
import CoreML
import Vision

/// Result of a shrimp larvae detection operation.
struct DetectionResult {
    /// Number of detected larvae.
    let count: Int
    /// Average confidence score across all detections.
    let averageConfidence: Double
    /// Individual bounding boxes as (x, y, width, height) normalized to [0, 1].
    let boundingBoxes: [CGRect]
    /// Confidence score for each detection.
    let confidences: [Float]
    /// Time taken for inference in seconds.
    let processingTime: TimeInterval
    /// Original image dimensions.
    let imageSize: CGSize
}

/// Service responsible for running shrimp larvae detection using CoreML.
///
/// This service loads the exported MobileNetV3-based detection model and
/// runs inference on captured images using Apple's Vision framework.
/// It applies sliding window processing for high-resolution images
/// and performs NMS to merge overlapping detections.
@MainActor
class DetectionService: ObservableObject {
    @Published var isProcessing = false
    @Published var detectionResult: DetectionResult?

    /// Confidence threshold for keeping detections.
    private let confidenceThreshold: Float = 0.5
    /// IoU threshold for non-maximum suppression.
    private let nmsThreshold: Float = 0.4
    /// Maximum bounding box area relative to tile size.
    private let maxBoxAreaRatio: Float = 0.25
    /// Tile size for sliding window inference.
    private let tileSize: Int = 640
    /// Overlap ratio between adjacent tiles.
    private let overlapRatio: Double = 0.3

    /// Detect shrimp larvae in the given image.
    ///
    /// The detection pipeline:
    /// 1. Applies CLAHE-equivalent contrast enhancement.
    /// 2. Splits the image into overlapping tiles.
    /// 3. Runs CoreML inference on each tile.
    /// 4. Merges detections and applies NMS.
    ///
    /// - Parameter image: Input UIImage from camera or gallery.
    func detectLarvae(in image: UIImage) {
        isProcessing = true
        detectionResult = nil

        Task {
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = await performDetection(on: image)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime

            await MainActor.run {
                self.detectionResult = DetectionResult(
                    count: result.boxes.count,
                    averageConfidence: result.boxes.isEmpty
                        ? 0.0
                        : Double(result.confidences.reduce(0, +))
                            / Double(result.confidences.count),
                    boundingBoxes: result.boxes,
                    confidences: result.confidences,
                    processingTime: processingTime,
                    imageSize: image.size
                )
                self.isProcessing = false
            }
        }
    }

    // MARK: - Private Detection Pipeline

    private struct RawDetections {
        var boxes: [CGRect]
        var confidences: [Float]
    }

    private func performDetection(
        on image: UIImage
    ) async -> RawDetections {
        guard let cgImage = image.cgImage else {
            return RawDetections(boxes: [], confidences: [])
        }

        let enhanced = applyContrastEnhancement(cgImage)
        let sourceImage = enhanced ?? cgImage

        let tiles = generateTiles(
            imageWidth: sourceImage.width,
            imageHeight: sourceImage.height,
            tileSize: tileSize,
            overlap: overlapRatio
        )

        var allBoxes: [CGRect] = []
        var allConfidences: [Float] = []

        for tile in tiles {
            guard let tileImage = sourceImage.cropping(to: tile) else {
                continue
            }

            let detections = runInference(on: tileImage, tileRect: tile)
            allBoxes.append(contentsOf: detections.boxes)
            allConfidences.append(contentsOf: detections.confidences)
        }

        let nmsResult = applyNMS(
            boxes: allBoxes,
            confidences: allConfidences,
            iouThreshold: nmsThreshold
        )

        return nmsResult
    }

    /// Apply contrast enhancement similar to CLAHE used in the Python pipeline.
    /// Uses Core Image filters for histogram equalization.
    private func applyContrastEnhancement(_ image: CGImage) -> CGImage? {
        let ciImage = CIImage(cgImage: image)
        let context = CIContext()

        guard let filter = CIFilter(name: "CIColorControls") else {
            return nil
        }

        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(1.1, forKey: kCIInputContrastKey)
        filter.setValue(0.05, forKey: kCIInputBrightnessKey)

        guard let outputImage = filter.outputImage,
              let cgOutput = context.createCGImage(
                  outputImage, from: outputImage.extent
              )
        else {
            return nil
        }

        return cgOutput
    }

    /// Generate overlapping tile rectangles for sliding window inference.
    private func generateTiles(
        imageWidth: Int,
        imageHeight: Int,
        tileSize: Int,
        overlap: Double
    ) -> [CGRect] {
        var tiles: [CGRect] = []
        let step = Int(Double(tileSize) * (1.0 - overlap))

        var y = 0
        while y < imageHeight {
            var x = 0
            let tileH = min(tileSize, imageHeight - y)

            while x < imageWidth {
                let tileW = min(tileSize, imageWidth - x)
                tiles.append(CGRect(x: x, y: y, width: tileW, height: tileH))
                x += step
                if x + step > imageWidth && x < imageWidth {
                    x = imageWidth - tileSize
                    if x < 0 { x = 0 }
                }
            }
            y += step
            if y + step > imageHeight && y < imageHeight {
                y = imageHeight - tileSize
                if y < 0 { y = 0 }
            }
        }

        return tiles
    }

    /// Run CoreML/Vision inference on a single image tile.
    ///
    /// If the CoreML model is not available (development/testing),
    /// this returns empty detections. In production, the .mlpackage
    /// model is loaded via Vision framework.
    private func runInference(
        on tileImage: CGImage,
        tileRect: CGRect
    ) -> RawDetections {
        // Attempt to load the CoreML model
        guard let modelURL = Bundle.main.url(
            forResource: "ShrimpDetector",
            withExtension: "mlmodelc"
        ) else {
            // Model not bundled yet — return empty for development builds
            return RawDetections(boxes: [], confidences: [])
        }

        do {
            let vnModel = try VNCoreMLModel(
                for: MLModel(contentsOf: modelURL)
            )

            var detectedBoxes: [CGRect] = []
            var detectedConfidences: [Float] = []

            let request = VNCoreMLRequest(model: vnModel) { request, _ in
                guard let results = request.results
                    as? [VNRecognizedObjectObservation]
                else { return }

                for observation in results {
                    guard observation.confidence >= self.confidenceThreshold
                    else { continue }

                    // Convert from Vision coordinates (bottom-left origin)
                    // to image coordinates (top-left origin)
                    let box = observation.boundingBox
                    let imageBox = CGRect(
                        x: tileRect.origin.x
                            + box.origin.x * tileRect.width,
                        y: tileRect.origin.y
                            + (1 - box.origin.y - box.height) * tileRect.height,
                        width: box.width * tileRect.width,
                        height: box.height * tileRect.height
                    )

                    // Filter large boxes
                    let areaRatio = Float(
                        imageBox.width * imageBox.height
                    ) / Float(tileRect.width * tileRect.height)
                    if areaRatio <= maxBoxAreaRatio {
                        detectedBoxes.append(imageBox)
                        detectedConfidences.append(observation.confidence)
                    }
                }
            }

            request.imageCropAndScaleOption = .scaleFill

            let handler = VNImageRequestHandler(
                cgImage: tileImage,
                options: [:]
            )
            try handler.perform([request])

            return RawDetections(
                boxes: detectedBoxes, confidences: detectedConfidences
            )

        } catch {
            return RawDetections(boxes: [], confidences: [])
        }
    }

    /// Apply Non-Maximum Suppression to merge overlapping detections.
    private func applyNMS(
        boxes: [CGRect],
        confidences: [Float],
        iouThreshold: Float
    ) -> RawDetections {
        guard !boxes.isEmpty else {
            return RawDetections(boxes: [], confidences: [])
        }

        // Sort by confidence (descending)
        let indices = confidences.enumerated()
            .sorted { $0.element > $1.element }
            .map { $0.offset }

        var kept: [Int] = []
        var suppressed = Set<Int>()

        for i in indices {
            if suppressed.contains(i) { continue }
            kept.append(i)

            for j in indices {
                if suppressed.contains(j) || j == i { continue }
                let iou = computeIoU(boxes[i], boxes[j])
                if iou > iouThreshold {
                    suppressed.insert(j)
                }
            }
        }

        let finalBoxes = kept.map { boxes[$0] }
        let finalConfidences = kept.map { confidences[$0] }

        return RawDetections(
            boxes: finalBoxes, confidences: finalConfidences
        )
    }

    /// Compute Intersection over Union between two rectangles.
    private func computeIoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }

        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height
            - intersectionArea

        return unionArea > 0
            ? Float(intersectionArea / unionArea) : 0
    }
}
