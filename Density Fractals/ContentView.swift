//
//  ContentView.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    var timerEnabled = true
    @State var fractalImage: CGImage?
    @State var mult = 1

    let timer = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack {
            ScrollView([.horizontal, .vertical]) {
                if let fractalImage = fractalImage {
                    Image(decorative: fractalImage, scale: 2)
                }
            }
            HStack {
                Text("huh")
            }
            Button("Save", action: saveImage)
        }
        .padding()
        .onReceive(timer) { time in
            if !timerEnabled {
                timer.upstream.connect().cancel()
            }

            for fractal in FractalRenderer.pauls {
                Task.detached(priority: .medium) { await fractal.orbit() }
            }

            Task {
                fractalImage = await FractalRenderer.grid.renderImage()
                print("Updated image at \(time)")
            }
        }
    }

    func saveImage() {
        Task {
            print("Saving!")
            let fractalImage = await FractalRenderer.grid.renderImage()
            let imageURL = URL.temporaryDirectory.appendingPathComponent("fractal-" + UUID().uuidString + ".png")
            let imageDest = CGImageDestinationCreateWithURL(
                imageURL as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(imageDest, fractalImage, nil)
            CGImageDestinationFinalize(imageDest)
            NSWorkspace.shared.activateFileViewerSelecting([imageURL])
        }
    }
}

#Preview {
    return ContentView(timerEnabled: false)
}


extension Sequence {
    func reduceIntoFirst(
        _ updateAccumulatingResult: (inout Element, Self.Element) throws -> ()
    ) rethrows -> Element? {
        var iter = makeIterator()
        guard var result = iter.next() else {
            return nil
        }
        for elem in dropFirst() {
            try updateAccumulatingResult(&result, elem)
        }
        return result
    }
}

