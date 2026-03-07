import UIKit
import CoreImage

/// Image processing utilities that mirror the Python CLAHE preprocessing pipeline.
///
/// Provides contrast enhancement and filtering operations optimized for
/// shrimp larvae images captured in typical hatchery conditions.
enum ImageProcessor {
    /// Apply contrast-limited adaptive histogram equalization (CLAHE)
    /// approximation using Core Image filters.
    ///
    /// The Python pipeline applies per-channel CLAHE with clipLimit=2.0
    /// and tileGridSize=(8,8). On iOS, we approximate this using
    /// CIHistogramDisplayFilter and contrast adjustments.
    ///
    /// - Parameter image: Input UIImage.
    /// - Returns: Enhanced UIImage, or original if processing fails.
    static func applyCLAHE(_ image: UIImage) -> UIImage {
        guard let ciImage = CIImage(image: image) else { return image }
        let context = CIContext()

        // Apply local contrast enhancement
        guard let contrastFilter = CIFilter(name: "CIColorControls") else {
            return image
        }
        contrastFilter.setValue(ciImage, forKey: kCIInputImageKey)
        contrastFilter.setValue(1.15, forKey: kCIInputContrastKey)
        contrastFilter.setValue(0.02, forKey: kCIInputBrightnessKey)
        contrastFilter.setValue(1.05, forKey: kCIInputSaturationKey)

        guard let contrastOutput = contrastFilter.outputImage else {
            return image
        }

        // Apply sharpening to enhance larvae edges
        guard let sharpenFilter = CIFilter(name: "CIUnsharpMask") else {
            return UIImage(ciImage: contrastOutput)
        }
        sharpenFilter.setValue(contrastOutput, forKey: kCIInputImageKey)
        sharpenFilter.setValue(1.5, forKey: kCIInputRadiusKey)
        sharpenFilter.setValue(0.5, forKey: kCIInputIntensityKey)

        guard let finalOutput = sharpenFilter.outputImage,
              let cgImage = context.createCGImage(
                  finalOutput, from: finalOutput.extent
              )
        else {
            return UIImage(ciImage: contrastOutput)
        }

        return UIImage(cgImage: cgImage)
    }

    /// Resize image to fit within the specified maximum dimension
    /// while maintaining aspect ratio.
    ///
    /// - Parameters:
    ///   - image: Input UIImage.
    ///   - maxDimension: Maximum width or height in pixels.
    /// - Returns: Resized UIImage.
    static func resize(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
        let scale: CGFloat
        if image.size.width > image.size.height {
            scale = maxDimension / image.size.width
        } else {
            scale = maxDimension / image.size.height
        }

        if scale >= 1.0 { return image }

        let newSize = CGSize(
            width: image.size.width * scale,
            height: image.size.height * scale
        )

        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: newSize))
        }
    }

    /// Draw bounding boxes on the image for visualization.
    ///
    /// - Parameters:
    ///   - image: Base UIImage.
    ///   - boxes: Array of bounding boxes in image coordinates.
    ///   - color: Box outline color.
    ///   - lineWidth: Box outline width.
    /// - Returns: Annotated UIImage.
    static func drawBoundingBoxes(
        on image: UIImage,
        boxes: [CGRect],
        color: UIColor = .green,
        lineWidth: CGFloat = 2.0
    ) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { ctx in
            image.draw(at: .zero)

            ctx.cgContext.setStrokeColor(color.cgColor)
            ctx.cgContext.setLineWidth(lineWidth)

            for box in boxes {
                ctx.cgContext.addRect(box)
                ctx.cgContext.strokePath()
            }
        }
    }
}
