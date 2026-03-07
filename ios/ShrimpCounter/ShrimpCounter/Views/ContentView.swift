import SwiftUI

/// Main content view providing navigation between camera capture and
/// image picker modes for shrimp larvae counting.
struct ContentView: View {
    @StateObject private var detectionService = DetectionService()
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var selectedImage: UIImage?
    @State private var showResults = false

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                headerSection
                imagePreviewSection
                actionButtons
                Spacer()
            }
            .padding()
            .navigationTitle("Shrimp Counter")
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(image: $selectedImage)
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraView(image: $selectedImage)
            }
            .onChange(of: selectedImage) { newImage in
                if let image = newImage {
                    processImage(image)
                }
            }
        }
    }

    // MARK: - View Components

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 60))
                .foregroundColor(.blue)

            Text("Shrimp Larvae Counter")
                .font(.title2)
                .fontWeight(.bold)

            Text("Capture or select an image to count shrimp larvae")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.top, 20)
    }

    private var imagePreviewSection: some View {
        Group {
            if let image = selectedImage {
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue.opacity(0.3), lineWidth: 2)
                        )

                    if detectionService.isProcessing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .padding(8)
                            .background(Color.black.opacity(0.6))
                            .cornerRadius(8)
                            .padding(8)
                    }
                }
                .frame(maxHeight: 300)

                if let result = detectionService.detectionResult {
                    ResultsOverlayView(result: result)
                }
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.gray.opacity(0.1))
                    .frame(height: 200)
                    .overlay(
                        VStack {
                            Image(systemName: "photo.on.rectangle")
                                .font(.system(size: 40))
                                .foregroundColor(.gray)
                            Text("No image selected")
                                .foregroundColor(.gray)
                        }
                    )
            }
        }
    }

    private var actionButtons: some View {
        HStack(spacing: 16) {
            Button(action: { showCamera = true }) {
                Label("Camera", systemImage: "camera.fill")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }

            Button(action: { showImagePicker = true }) {
                Label("Gallery", systemImage: "photo.fill")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
        }
    }

    // MARK: - Actions

    private func processImage(_ image: UIImage) {
        detectionService.detectLarvae(in: image)
    }
}

/// Displays detection results with count and confidence information.
struct ResultsOverlayView: View {
    let result: DetectionResult

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Larvae Count")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(result.count)")
                        .font(.system(size: 36, weight: .bold))
                        .foregroundColor(.blue)
                }

                Spacer()

                VStack(alignment: .trailing) {
                    Text("Avg Confidence")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f%%", result.averageConfidence * 100))
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.green)
                }
            }

            HStack {
                Label(
                    String(format: "%.2fs", result.processingTime),
                    systemImage: "clock"
                )
                .font(.caption)
                .foregroundColor(.secondary)

                Spacer()

                Label(
                    "\(result.imageSize.width)×\(result.imageSize.height)",
                    systemImage: "aspectratio"
                )
                .font(.caption)
                .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}
