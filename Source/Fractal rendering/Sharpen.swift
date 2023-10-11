//
//  Sharpen.swift
//  Density Fractals
//
//  Created by Paul on 10/9/23.
//

import Foundation
import MetalPerformanceShaders

struct ConvolutionKernel {
    var width, height: Int
    var weights: [Float]

    func makeMPSKernel(device: any MTLDevice) -> MPSImageConvolution {
        if width * height != weights.count {
            fatalError("ConvolutionKernel size \(width)x\(height) does not match weight count of \(weights.count)")
        }
        return MPSImageConvolution(device: device, kernelWidth: width, kernelHeight: height, weights: weights)
    }

    static func sharpen(amount: Float, sigma: Float) -> ConvolutionKernel {
        let size = Int(ceil(sigma * 2)) * 2 + 1
        let center = size / 2
        let gaussScale = 1.0 / (sigma * sigma * 2 * .pi)

        var weights: [Float] = []
        weights.reserveCapacity(size * size)
        for x in 0..<size {
            for y in 0..<size {
                let r = hypot(Float(x - center), Float(y - center))
                weights.append(-amount * gaussScale * exp(-r * r / (2 * sigma * sigma)))
            }
        }
        weights[center + size * center] += 1 + amount
        return ConvolutionKernel(width: size, height: size, weights: weights)
    }
}
