//
//  FractalRenderer.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import Foundation

actor FractalRenderer {
    static let pauls = (1...8).map { _ in
        FractalRenderer(size: 2048, rotation: 1.6620565029879701, thetaOffset: 3.5144442745012823)
    }

    let rotation: Double
    let thetaOffset: Double

    private(set) var grid: CountGrid<UInt64>

    private var running = false

    init(size: Int, rotation: Double, thetaOffset: Double) {
        self.rotation = rotation
        self.thetaOffset = thetaOffset

        grid = CountGrid(width: size, height: size)
    }

    func orbit() async {
        guard !running else { return }
        running = true

        var x: Double = 0.5
        var y: Double = 0.0

        let sizeD = Double(grid.width)
        let cosRot = cos(rotation),
            sinRot = sin(rotation)

        var rand = SystemRandomNumberGenerator()

        var count = 0
        var timer = ContinuousClock.now

        while running {
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

                grid.touch(
                    Int((x / 2 + 0.5) * sizeD),
                    Int((y / 2 + 0.5) * sizeD)
                )

                randBits >>= 1
                count += 1
            }

            if count >= 10000000 {
                print(Double(count) / (ContinuousClock.now - timer).milliseconds, "orbits per ms")

                await Task.yield()

                count = 0
                timer = ContinuousClock.now
            }
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

