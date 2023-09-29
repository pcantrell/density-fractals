//
//  CountGrid.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import CoreGraphics

struct CountGrid<CountType: BinaryInteger> {
    let width, height: Int
    private var buffer: [CountType]
    private(set) var maxCount: CountType = 0

    init(width: Int, height: Int) {
        self.width = width
        self.height = height

        buffer = .init(repeating: 0, count: width * height)
    }

    mutating func touch(_ x: Int, _ y: Int) {
        guard isInBounds(x, y) else {
            return
        }
        let index = indexOf(x, y)
        buffer[index] += 1
        maxCount = max(maxCount, buffer[index])
    }

    subscript(_ x: Int, _ y: Int) -> CountType {
        precondition(
            isInBounds(x, y),
            "CountGrid indices out of bounds: [\(x), \(y)] not within [\(width), \(height)]")
        return buffer[indexOf(x, y)]
    }

    private func isInBounds(_ x: Int, _ y: Int) -> Bool {
        return x >= 0 && x < width && y >= 0 && y < width
    }

    private func indexOf(_ x: Int, _ y: Int) -> Int {
        return x + y * width
    }

    mutating func merge(_ other: CountGrid) {
        guard self.width == other.width && self.height == other.height else {
            fatalError("Cannot merge CountGrids with different sizes")
        }
        for i in buffer.indices {
            buffer[i] += other.buffer[i]
        }
        maxCount = buffer.max() ?? 0
    }

    func renderImage() -> CGImage {
        // on-demand provider: https://stackoverflow.com/a/2261343/239816
        print("Rendering image...")
        let numComponents = 3
        let numBytes = height * width * numComponents
        var pixelData = [UInt8](repeating: 0, count: numBytes)

        print("Filling image buffer...")
        let maxCountD = Double(maxCount)
        if maxCountD > 0 {
            for i in pixelData.indices {
                pixelData[i] = UInt8(Double(buffer[i / 3]) / maxCountD * 255)
            }
        }
        print("Done filling image buffer.")

        let colorspace = CGColorSpaceCreateDeviceRGB()
        let rgbData = CFDataCreate(nil, pixelData, numBytes)!
        let provider = CGDataProvider(data: rgbData)!
        print("Creating CGImage...")
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8 * numComponents,
            bytesPerRow: width * numComponents,
            space: colorspace,
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: CGColorRenderingIntent.defaultIntent)!
    }
}
