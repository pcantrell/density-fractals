//
//  MetalGrid.swift
//  Density Fractals
//
//  Created by Paul on 9/30/23.
//

import Foundation
import Metal
import Alloy
import IOKit.pwr_mgt
import MetalPerformanceShaders

private let logTiming = true, logStats = false

/// Manages the rendering of a density fractal on the GPU. This is a stateful actor: it hangs on to
/// a current set of shape and color parameters, and a most recently rendered image. Rendering or
/// animating updates its state; to render mutliple images independently, use multiple instances.
///
actor MetalFractalRenderer {
    // The size in pixels of the output image
    let size: Int

    // Transform parameters, which determine the current shape of the fractals
    var rotation: Double
    var thetaOffset: Double

    // The colorization parameters, which control the int density → RGB color mapping
    var colorScheme: FractalColorScheme

    // Color parameter change over time
    private var
        coolHue = Wave(Δphase:  pow(0.0030, 0.99)),  // exponents make speeds all mutually irrational
        coolSat = Wave(Δphase:  pow(0.0070, 0.99), phase: -0.25),
        medR    = Wave(Δphase:  pow(0.0014, 0.99), range: -0.1...0.5),
        medG    = Wave(Δphase: -pow(0.0025, 0.99), range: -0.1...0.5),
        medB    = Wave(Δphase:  pow(0.0036, 0.99), range: -0.1...0.5),
        hotHue  = Wave(Δphase: -pow(0.0053, 0.99)),
        hotSat  = Wave(Δphase:  pow(0.0042, 0.99), phase: -0.25, range: 0.2...0.7)

    // Coloring depends on max density, which can vary both suddenly with small param changes
    // and randomly across multiple renders. Smoothing max density prevents flickering.
    private var maxDensitySmoothing = GaussianMedian(windowSize: 9, sigma: 1.6)

    // Preallocated metal resources
    private let metal: MTLContext
    private var lastCompletedDensityBuf: (any MTLBuffer)?
    private let renderOrbitPipelineState: any MTLComputePipelineState
    private let maxDensityPipelineState: any MTLComputePipelineState
    private let totalDensityPipelineState: any MTLComputePipelineState
    private let renderSquarePipelineState: any MTLRenderPipelineState
    private let squareVertexBuf: any MTLBuffer

    // Internal tuning params
    private let maxPointBatchPerThread = 10_000
    private let gpuThreadCount = 10_000

    private let pixelFormat = MTLPixelFormat.bgra8Unorm

    // Allows a single listener to receive notifcations whenever a new image is rendered
    private var frameRenderCallback: @MainActor @Sendable () -> Void = { }

    init(size: Int, rotation: Double = 0, thetaOffset: Double = 0) {
        self.size = size
        self.rotation = rotation
        self.thetaOffset = thetaOffset

        self.colorScheme = FractalColorScheme(
            cool:   simd_float3.zero,
            medium: simd_float3.zero,
            hot:    simd_float3.zero)

        metal = try! MTLContext(device: MTLCreateSystemDefaultDevice()!)
        let metalLib = try! metal.library(for: .main)

        renderOrbitPipelineState = try! metalLib.computePipelineState(function: "renderOrbit")
        maxDensityPipelineState = try! metalLib.computePipelineState(function: "maxDensity")
        totalDensityPipelineState = try! metalLib.computePipelineState(function: "totalDensity")

        let renderSquareDescriptor = MTLRenderPipelineDescriptor()
        renderSquareDescriptor.vertexFunction = metalLib.makeFunction(name: "densityVertex")!
        renderSquareDescriptor.fragmentFunction = metalLib.makeFunction(name: "densityFragment")!
        renderSquareDescriptor.colorAttachments[0].pixelFormat = pixelFormat
        renderSquarePipelineState = try! metal.renderPipelineState(descriptor: renderSquareDescriptor)

        let squareVertices = [
            vector_float2(0, 0),
            vector_float2(0, 1),
            vector_float2(1, 0),
            vector_float2(1, 1),
        ]
        squareVertexBuf = try! metal.buffer(with: squareVertices, options: .cpuCacheModeWriteCombined)
    }

    // MARK: Fractal computation

    func renderImage(samplePoints: Int) async -> any MTLBuffer {
        let timer = ContinuousClock.now
        defer {
            if logTiming {
                print("Rendered image in \(ContinuousClock.now - timer)")
            }
        }

        let densityBuf = try! metal.buffer(for: DensityCount.self, count: size * size, options: .storageModePrivate)

        var pointsToRender = samplePoints
        while pointsToRender > 0 {
            pointsToRender -= renderOrbit(to: densityBuf, maxPoints: pointsToRender)
            await Task.yield()  // In case anybody's showing updates during rendering
        }

        lastCompletedDensityBuf = densityBuf
        return densityBuf
    }

    /// Render a batch of points, updating hits counts in `densityBuf` and returning count of
    /// actual points rendered.
    ///
    private func renderOrbit(to densityBuf: any MTLBuffer, maxPoints: Int = .max) -> Int {
        let idealPointBatchSize = min(maxPointBatchPerThread * gpuThreadCount, maxPoints)
        let pointBatchPerThread = (idealPointBatchSize + gpuThreadCount - 1) / gpuThreadCount
        let pointBatchSize = pointBatchPerThread * gpuThreadCount

        let timer = ContinuousClock.now
        defer {
            if logTiming {
                let elapsedTime = ContinuousClock.now - timer
                print("rendered \(pointBatchSize) points in \(elapsedTime)"
                    + " (\(Double(pointBatchSize) / elapsedTime.seconds / 1_000_000) megapoints/sec)")
            }
        }

        try! metal.scheduleAndWait { cmdBuffer in
            cmdBuffer.compute { cmdEncoder in
                cmdEncoder.setValue(
                    FractalShaderParams(
                        rotation: Float(rotation),
                        thetaOffset: Float(thetaOffset),
                        gridSize: Int32(size),
                        pointBatchPerThread: Int32(pointBatchPerThread),
                        gpuThreadCount: Int32(gpuThreadCount),
                        randSeed: .random(in: .fullRange)),
                    at: 0)
                cmdEncoder.setBuffer(densityBuf, offset: 0, index: 1)
                cmdEncoder.dispatch1d(state: renderOrbitPipelineState, exactly: gpuThreadCount)
            }
        }

        return pointBatchSize
    }

    private func computeMaxDensity(in densityBuf: any MTLBuffer) -> DensityCount {
        computeChunked(densityBuf: densityBuf, pipelineState: maxDensityPipelineState)
            .max() ?? 0
    }

    private func computeTotalDensity(of densityBuf: any MTLBuffer) -> UInt64 {
        computeChunked(densityBuf: densityBuf, pipelineState: totalDensityPipelineState)
            .reduce(0, +)
    }

    private func computeChunked<Result>(
        densityBuf: any MTLBuffer,
        pipelineState: any MTLComputePipelineState
    ) -> [Result] {
        let timer = ContinuousClock.now
        defer {
            if logTiming {
                print("Computed stats in \(ContinuousClock.now - timer)")
            }
        }

        let chunkCount = 1000
        let resultBuf = try! metal.buffer(for: Result.self, count: chunkCount, options: [])

        try! metal.scheduleAndWait { cmdBuffer in
            cmdBuffer.compute { cmdEncoder in
                cmdEncoder.setValue(size, at: 0)
                cmdEncoder.setBuffer(densityBuf, offset: 0, index: 1)
                cmdEncoder.setBuffer(resultBuf, offset: 0, index: 2)
                let chunkSize = (Int(size * size) + chunkCount - 1) / chunkCount
                cmdEncoder.setValue(chunkSize, at: 3)
                cmdEncoder.dispatch1d(state: pipelineState, exactly: chunkCount)
            }
        }

        var result: [Result] = []
        resultBuf.contents().withMemoryRebound(to: Result.self, capacity: 1) { data in
            for i in 0..<chunkCount {
                result.append(data[i])
            }
        }
        return result
    }

    // MARK: Drawing

    func drawLastCompleted(descriptor: MTLRenderPassDescriptor) async -> Bool {
        guard let densityBuf = lastCompletedDensityBuf else {
            return false
        }
        await draw(densityBuf, descriptor: descriptor)
        return true
    }

    @discardableResult
    func draw(
        _ densityBuf: any MTLBuffer,
        descriptor: MTLRenderPassDescriptor,
        copyback: MTLTexture? = nil,  // to make pixels available to CPU
        awaitCompletion: Bool = false
    ) async -> DensityStats {
        let gridPixelCount = pow(Float(size), 2)
        let maxDensity = computeMaxDensity(in: densityBuf)
        let totalDensity = computeTotalDensity(of: densityBuf)
        let totalDensityScaled = Float(totalDensity) / gridPixelCount * 42
        maxDensitySmoothing.append(Double(maxDensity))
        let maxDensitySmoothed = Float(ceil(maxDensitySmoothing.average()))

        // Create intermediate texture for rendering before sharpening
        let descriptor = descriptor.copy() as! MTLRenderPassDescriptor
        let finalOutputTexture = descriptor.colorAttachments[0].texture!
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: size,
            height: size,
            mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]
        textureDescriptor.storageMode = .managed
        let rawFractalTexture = try! metal.texture(descriptor: textureDescriptor)
        descriptor.colorAttachments[0].texture = rawFractalTexture

        try! metal.schedule(wait: awaitCompletion) { cmdBuffer in
            // Render fractal
            cmdBuffer.render(descriptor: descriptor) { cmdEncoder in
                cmdEncoder.setRenderPipelineState(renderSquarePipelineState)
                cmdEncoder.setVertexBuffer(squareVertexBuf, offset: 0, index: 0)
                cmdEncoder.setFragmentBuffer(densityBuf, offset: 0, index: 0)
                cmdEncoder.setFragmentValue(size, at: 1)
                cmdEncoder.setFragmentValue(maxDensitySmoothed, at: 2)
                cmdEncoder.setFragmentValue(totalDensityScaled, at: 3)
                cmdEncoder.setFragmentValue(colorScheme, at: 4)
                cmdEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }

            // Sharpen image
            ConvolutionKernel.sharpen(amount: 0.7, sigma: Float(size) * 0.008)
                .makeMPSKernel(device: metal.device)
                .encode(
                    commandBuffer: cmdBuffer,
                    sourceTexture: rawFractalTexture,
                    destinationTexture: finalOutputTexture)

            // Make pixels accessible to CPU if needed (for video output, but not for window update)
            if let copyback = copyback {
                cmdBuffer.blit { copybackEncoder in
                    copybackEncoder.synchronize(resource: copyback)
                }
            }
        }

        return DensityStats(maxDensity: maxDensity, totalDensity: totalDensity)
    }

    // MARK: Animation

    func renderAnimation(
        startTime: TimeInterval = 0,
        duration: TimeInterval,
        speed: Double,
        apogeeSlowdown: Double = 1,
        frameRate: Int,
        pointsPerFrame: Int,
        ΔrotationPerSecond: Double,
        ΔthetaOffsetPerSecond: Double
    ) async {
        var assertionID: IOPMAssertionID = 0
        IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "Rendering fractal" as CFString,
            &assertionID)
        defer {
            IOPMAssertionRelease(assertionID)
        }

        let encoder = try! VideoEncoder(width: size, height: size, frameRate: frameRate)
        print("Generating video at: \(encoder.output.path)")

        updateColorScheme(Δt: 0)  // initialize colors

        var timeRendered: Double = -startTime
        let ΔtBase = speed / Double(frameRate)

        while timeRendered < duration {
            let Δt = ΔtBase / (1 + apogeeSlowdown * (1 * pow((1 - cos(rotation)) / 2, 80)))

            if timeRendered >= 0 {
                print(
                    "Rendering \(timeRendered) / \(duration) sec",
                    "(\(Int(timeRendered / duration * 100))%)",
                    "t=\(startTime + timeRendered))...")

                let densityBuf = await renderImage(samplePoints: pointsPerFrame)

                // Show on screen
                Task {
                    await MainActor.run(body: self.frameRenderCallback)
                }

                let stats = await self.writeVideoFrame(from: densityBuf, to: encoder)

                if logStats {
                    print("           totalDensity: \(stats.totalDensity)")
                    print("             maxDensity: \(stats.maxDensity)")
                    print("          concentration: \(stats.concentration)")
                }
                if logStats || logTiming {
                    print()
                }
            }

            updateColorScheme(Δt: Δt)

            thetaOffset += ΔthetaOffsetPerSecond * Δt
            rotation += ΔrotationPerSecond * Δt

            timeRendered += 1 / Double(frameRate)
        }

        await encoder.end()
        print("Rendering complete.")
        print(encoder.output.path)
        try! print(FileManager.default.attributesOfItem(atPath: encoder.output.path)[.size] ?? "???", "bytes")
    }

    private func updateColorScheme(Δt: Double) {
        colorScheme.cool = simdColor(h: coolHue.next(speed: Δt), s: coolSat.next(speed: Δt), b: 0.6)
        colorScheme.medium = simdColor(r: medR.next(speed: Δt), g: medG.next(speed: Δt), b: medB.next(speed: Δt))
        colorScheme.hot = simdColor(h: hotHue.next(speed: Δt), s: hotSat.next(speed: Δt), b: 0.9) * 2 - 1
    }

    private func writeVideoFrame(from densityBuf: any MTLBuffer, to encoder: VideoEncoder) async -> DensityStats {
        // Many portions adapted from https://github.com/warrenm/MetalOfflineRecording/blob/b4ebaddc37950fd5d835ed60530e7f1905e6d293/MetalOfflineRecording/Renderer.swift
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: size,
            height: size,
            mipmapped: false)
        textureDescriptor.usage = [.shaderWrite, .renderTarget]
        textureDescriptor.storageMode = .managed
        let outputTexture = try! metal.texture(descriptor: textureDescriptor)

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = outputTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.5, 0.5, 1)

        let stats = await draw(densityBuf, descriptor: renderPassDescriptor, copyback: outputTexture, awaitCompletion: true)
        try! await encoder.writeFrame(forTexture: outputTexture)
        return stats
    }

    func onFrameRendered(action: @escaping @MainActor @Sendable () -> Void) {
        frameRenderCallback = action
    }
}

typealias PointIndex = Int
typealias DensityCount = UInt32

struct DensityStats {
    var maxDensity: DensityCount
    var totalDensity: UInt64

    var concentration: Double {
        Double(maxDensity) / Double(totalDensity)
    }
}

private struct Wave {
    var Δphase: Double
    var phase: Double = 0
    var range: ClosedRange<Double> = 0...1

    mutating func next(speed: Double = 1) -> Double {
        defer {
            phase = (phase + Δphase * speed).remainder(dividingBy: 1)
        }
        return (sin(phase * 2 * .pi) / 2 + 0.5)
            * (range.upperBound - range.lowerBound) + range.lowerBound
    }
}

/// A hybrid of mean and median: computes the gaussian-weighted average of values in sorted order.
///
private struct GaussianMedian {
    private let windowSize: Int
    private let sigma: Double
    private var recentValues: [Double] = []

    init(windowSize: Int, sigma: Double) {
        self.windowSize = windowSize
        self.sigma = sigma
    }

    mutating func clear() {
        recentValues.removeAll(keepingCapacity: true)
    }

    mutating func append(_ value: Double) {
        recentValues.append(value)
        if recentValues.count > windowSize {
            recentValues.removeFirst()
        }
    }

    func average() -> Double {
        var total: Double = 0
        var totalWeight: Double = 0
        let center = Double(windowSize - 1) / 2
        for (index, value) in recentValues.sorted().enumerated() {
            let weight = exp(-pow(Double(index) - center, 2) / (2 * sigma * sigma))
            total += value * weight
            totalWeight += weight
        }
        return total / totalWeight
    }
}

private func simdColor(h: CGFloat, s: CGFloat, b: CGFloat) -> simd_float3 {
    let color = NSColor(hue: h, saturation: s, brightness: b, alpha: 1)
    var r: CGFloat = 0,
        g: CGFloat = 0,
        b: CGFloat = 0
    color.getRed(&r, green: &g, blue: &b, alpha: nil)
    return simdColor(r: r, g: g, b: b)
}

private func simdColor(r: Double, g: Double, b: Double) -> simd_float3 {
    return simd_float3(Float(r), Float(g), Float(b))
}
