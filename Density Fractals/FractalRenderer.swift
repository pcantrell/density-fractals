//
//  FractalRenderer.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import Foundation

private let pointBatchSize = 10000000 //16000000

typealias Count = UInt32

actor FractalRenderer {
    static let grid = CountGrid<Count>(size: 2048)
    static let pauls = (1...8).map { _ in
        FractalRenderer(destination: grid, rotation: 1.6620565029879701, thetaOffset: 3.5144442745012823)
    }

    let rotation: Double
    let thetaOffset: Double

    let destination: CountGrid<Count>

    private var running = false

    init(destination: CountGrid<Count>, rotation: Double, thetaOffset: Double) {
        self.destination = destination
        self.rotation = rotation
        self.thetaOffset = thetaOffset
    }

    func orbit() async {
        guard !running else { return }
        running = true

        var x: Double = 0.5
        var y: Double = 0.0

        let size = max(destination.width, destination.height)
        let sizeD = Double(size)
        let cosRot = cos(rotation),
            sinRot = sin(rotation)

        while running {
            var rand = SplitMix64()

//            let timer = ContinuousClock.now

            var points: [(Int, Int)] = []
            points.reserveCapacity(pointBatchSize)

            for _ in 0 ..< pointBatchSize / 64 {
                var randBits = rand.next()
                for _ in 0..<64 {
                    if (randBits & 1) == 0 {
                        (x, y) = (
                            x * cosRot + y * sinRot,
                            y * cosRot - x * sinRot
                        )
                    } else {
                        let r = x * 0.5 + 0.5
                        let theta = y * .pi + thetaOffset
                        x = r * cos(theta)
                        y = r * sin(theta)
                    }

                    points.append((
                        Int((x / 2 + 0.5) * sizeD),
                        Int((y / 2 + 0.5) * sizeD)
                    ))

                    randBits >>= 1
                }

//                if points.count >= pointBatchSize {
//                    points.removeAll(keepingCapacity: true)
//                }
            }

//print("will touch \(points.count)")

            await destination.touchAll(points)

//            await destination.addCount(points.count)

//print("did touch all")
//            await Task.yield()

//            let pointsBaked = points
//            Task.detached(priority: .medium) {
//                await self.destination.touchAll(pointsBaked)
//            }

//            await Task.yield()

//            print(Double(pointBatchSize) / (ContinuousClock.now - timer).milliseconds, "orbits per ms")
        }
    }

    func stop() {
        running = false
    }
}

extension Duration {
    var milliseconds: Double {
        self / .milliseconds(1)
    }
}


// From https://github.com/apple/swift/blob/main/benchmark/utils/TestsUtils.swift#L242-L263
//
// This is a fixed-increment version of Java 8's SplittableRandom generator.
// It is a very fast generator passing BigCrush, with 64 bits of state.
// See http://dx.doi.org/10.1145/2714064.2660195 and
// http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
//
// Derived from public domain C implementation by Sebastiano Vigna
// See http://xoshiro.di.unimi.it/splitmix64.c
//
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    public init() {
        self.init(seed: .random(in: .fullRange))
    }

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        self.state &+= 0x9e3779b97f4a7c15
        var z: UInt64 = self.state
        z = (z ^ (z &>> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z &>> 27)) &* 0x94d049bb133111eb
        return z ^ (z &>> 31)
    }
}

extension FixedWidthInteger {
    static var fullRange: ClosedRange<Self> {
        (.min)...(.max)
    }
}

extension ClosedRange where Bound: FixedWidthInteger {
    static var fullRange: Self {
        Bound.fullRange
    }
}
