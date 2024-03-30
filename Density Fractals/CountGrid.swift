//
//  CountGrid.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import CoreGraphics

actor CountGrid<CountType: BinaryInteger> {
    let width, height: Int
    private var buffer: [CountType]
    private(set) var maxCount: CountType = 0

private var pointCount = 0
private var timer = ContinuousClock.now

    init(size: Int) {
        self.init(width: size, height: size)
    }

    init(width: Int, height: Int) {
        self.width = width
        self.height = height

        buffer = .init(repeating: 0, count: width * height)
    }

    func clear() {
        maxCount = 0
        for index in buffer.indices {
            buffer[index] = 0
        }
    }

    func touch(_ x: Int, _ y: Int) {
        guard isInBounds(x, y) else {
            return
        }
        let index = indexOf(x, y)
        buffer[index] += 1
        maxCount = max(maxCount, buffer[index])
    }

    func touchAll(_ points: [(x: Int, y: Int)]) {
//print("Touching points")
//let touchTimer = ContinuousClock.now
        for point in points {
            touch(point.x, point.y)
        }
        pointCount += points.count
//print("Touched \(points.count) points in \(ContinuousClock.now - touchTimer)")
    }

func addCount(_ count: Int) {
pointCount += count
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

    func renderImage() -> (image: CGImage, maxCount: CountType) {
        print("Rendering image...")
        print(Double(pointCount) / (ContinuousClock.now - timer).milliseconds, "orbits per ms")
        defer {
            pointCount = 0
            timer = ContinuousClock.now
            print("Done rendering.")
        }

        // on-demand provider: https://stackoverflow.com/a/2261343/239816
        let numComponents = 3
        let numBytes = height * width * numComponents
        var pixelData = [UInt8](repeating: 0, count: numBytes)

//        print("Filling image buffer...")
print("image quality: \(maxCount)")
        let maxCountD = Double(maxCount)
        if maxCountD > 0 {
            for i in pixelData.indices {
                pixelData[i] = UInt8(pow(Double(buffer[i / 3]) / maxCountD, 0.6) * 255)
            }
        }
//        print("Done filling image buffer.")

        let colorspace = CGColorSpaceCreateDeviceRGB()
        let rgbData = CFDataCreate(nil, pixelData, numBytes)!
        let provider = CGDataProvider(data: rgbData)!
//        print("Creating CGImage...")
        let image = CGImage(
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
        return (image, maxCount)
    }
}
