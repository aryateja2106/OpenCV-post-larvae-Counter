import SwiftUI

/// Main entry point for the ShrimpCounter iOS application.
///
/// The app provides real-time shrimp larvae counting using the device camera
/// and a CoreML-based MobileNetV3 detection model. Designed for use by
/// hatchery operators and shrimp seed farmers.
@main
struct ShrimpCounterApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
