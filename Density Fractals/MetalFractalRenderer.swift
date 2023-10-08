//
//  MetalGrid.swift
//  Density Fractals
//
//  Created by Paul on 9/30/23.
//

import Foundation
import Metal
import MetalKit
import SwiftUI
import IOKit.pwr_mgt

typealias PointIndex = Int
typealias DensityCount = UInt32

struct DensityStats {
    var maxDensity: DensityCount
    var totalDensity: UInt64

    var concentration: Double {
        Double(maxDensity) / Double(totalDensity)
    }
}

private let logTiming = true, logStats = false

struct MetalFractalView: NSViewRepresentable {
    let renderer: MetalFractalRenderer

    // From https://blog.canopas.com/how-to-get-started-with-metal-apis-with-uiview-and-swiftui-124643d8209e#67e6
    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.enableSetNeedsDisplay = true
        view.colorspace = .init(name: CGColorSpace.displayP3)

        Task {
            await renderer.onFrameRendered {
                view.needsDisplay = true
            }
        }

        return view
    }

    func updateNSView(_ uiView: MTKView, context: Context) {
        uiView.needsDisplay = true
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(metalGrid: renderer)
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: MTKView, context: Context) -> CGSize? {
        CGSize(width: renderer.size / 2, height: renderer.size / 2)
    }

    class Coordinator: NSObject, MTKViewDelegate {
        let metalGrid: MetalFractalRenderer

        init(metalGrid: MetalFractalRenderer) {
            self.metalGrid = metalGrid
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // We always render to fixed-size texture matching renderer size
        }

        func draw(in view: MTKView) {
            guard
                let descriptor = view.currentRenderPassDescriptor,
                let drawable = view.currentDrawable
            else {
                print("No drawable / descriptor (yet?)")
                return
            }
            Task {
                await metalGrid.draw(descriptor: descriptor)
                drawable.present()
            }
        }
    }
}

actor MetalFractalRenderer {
    let size: Int

    var rotation: Double
    var thetaOffset: Double

    var colorScheme: FractalColorScheme

    private var
        coolHue = Wave(Δphase:  pow(0.0030, 0.99)),  // exponents makes speeds all mutually irrational
        coolSat = Wave(Δphase:  pow(0.0070, 0.99), phase: -0.25),
        medR    = Wave(Δphase:  pow(0.0014, 0.99), range: 0...0.5),
        medG    = Wave(Δphase: -pow(0.0025, 0.99), range: 0...0.5),
        medB    = Wave(Δphase:  pow(0.0036, 0.99), range: 0...0.5),
        hotHue  = Wave(Δphase: -pow(0.0053, 0.99)),
        hotSat  = Wave(Δphase:  pow(0.0042, 0.99), phase: -0.25, range: 0.2...0.7)

    private let gpu: any MTLDevice
    private var densityBuf, lastCompletedDensityBuf: (any MTLBuffer)?
    private let cmdQueue: any MTLCommandQueue
    private let renderOrbitPipelineState: any MTLComputePipelineState
    private let maxDensityPipelineState: any MTLComputePipelineState
    private let totalDensityPipelineState: any MTLComputePipelineState
    private let renderSquarePipelineState: any MTLRenderPipelineState
    private let squareVertexBuf: any MTLBuffer

    private let maxPointBatchPerThread = 10_000
    private let gpuThreadCount = 10_000

    private let pixelFormat = MTLPixelFormat.bgra8Unorm

    private var frameRenderCallback: @MainActor @Sendable () -> Void = { }

    init(size: Int, rotation: Double = 0, thetaOffset: Double = 0) {
        self.size = size
        self.rotation = rotation
        self.thetaOffset = thetaOffset

        self.colorScheme = FractalColorScheme(
            cool:   simd_float3.zero,
            medium: simd_float3.zero,
            hot:    simd_float3.zero)

        gpu = MTLCreateSystemDefaultDevice()!
        let gpuLibrary = gpu.makeDefaultLibrary()!

        renderOrbitPipelineState = try! gpu.makeComputePipelineState(function:
            gpuLibrary.makeFunction(name: "renderOrbit")!)

        maxDensityPipelineState = try! gpu.makeComputePipelineState(function:
            gpuLibrary.makeFunction(name: "maxDensity")!)

        totalDensityPipelineState = try! gpu.makeComputePipelineState(function:
            gpuLibrary.makeFunction(name: "totalDensity")!)

        let renderSquareDescriptor = MTLRenderPipelineDescriptor()
        renderSquareDescriptor.vertexFunction = gpuLibrary.makeFunction(name: "densityVertex")!
        renderSquareDescriptor.fragmentFunction = gpuLibrary.makeFunction(name: "densityFragment")!
        renderSquareDescriptor.colorAttachments[0].pixelFormat = pixelFormat
        renderSquarePipelineState = try! gpu.makeRenderPipelineState(descriptor: renderSquareDescriptor)

        cmdQueue = gpu.makeCommandQueue()!

        let squareVertices = [
            RenderVertex(position: SIMD2(0, 0)),
            RenderVertex(position: SIMD2(0, 1)),
            RenderVertex(position: SIMD2(1, 0)),
            RenderVertex(position: SIMD2(1, 1)),
        ]
        squareVertexBuf = gpu.makeBuffer(
            bytes: squareVertices,
            length: squareVertices.count * MemoryLayout.stride(ofValue: squareVertices[0]),
            options: .cpuCacheModeWriteCombined)!
    }

    // Render a batch of points, returning count of actual points rendered
    private func renderOrbit(maxPoints: Int = .max) -> Int {
        guard let densityBuf = densityBuf else {
            fatalError("no densityBuf available for rendering")
        }

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

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(renderOrbitPipelineState)

        var params = FractalShaderParams(
            rotation: Float(rotation),
            thetaOffset: Float(thetaOffset),
            gridSize: Int32(size),
            pointBatchPerThread: Int32(pointBatchPerThread),
            gpuThreadCount: Int32(gpuThreadCount),
            randSeed: .random(in: .fullRange))
        cmdEncoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 0)

        cmdEncoder.setBuffer(densityBuf, offset: 0, index: 1)

        cmdEncoder.dispatchThreads(
            MTLSize(width: gpuThreadCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(gpuThreadCount, renderOrbitPipelineState.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))

        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

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

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(pipelineState)

        var size = self.size
        cmdEncoder.setBytes(&size, length: MemoryLayout.size(ofValue: size), index: 0)

        cmdEncoder.setBuffer(densityBuf, offset: 0, index: 1)

        let chunkCount = 1000
        let resultBuf = gpu.makeBuffer(length: MemoryLayout<Result>.stride * chunkCount)!
        cmdEncoder.setBuffer(resultBuf, offset: 0, index: 2)

        var chunkSize = (Int(size * size) + chunkCount - 1) / chunkCount
        cmdEncoder.setBytes(&chunkSize, length: MemoryLayout.size(ofValue: chunkSize), index: 3)

        cmdEncoder.dispatchThreads(
            MTLSize(width: chunkCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Int(ceil(sqrt(Float(chunkCount)))), renderOrbitPipelineState.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))

        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        var result: [Result] = []
        resultBuf.contents().withMemoryRebound(to: Result.self, capacity: 1) { data in
            for i in 0..<chunkCount {
                result.append(data[i])
            }
        }
        return result
    }

    @discardableResult
    func draw(
        descriptor: MTLRenderPassDescriptor,
        copyback: MTLTexture? = nil,  // to make pixels available to CPU
        awaitCompletion: Bool = false
    ) async -> DensityStats {
        guard let densityBuf = lastCompletedDensityBuf ?? densityBuf else {
            return DensityStats(maxDensity: 0, totalDensity: 0)
        }

        let maxDensity = computeMaxDensity(in: densityBuf)
        let totalDensity = computeTotalDensity(of: densityBuf)

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        cmdEncoder.setRenderPipelineState(renderSquarePipelineState)

        cmdEncoder.setVertexBuffer(squareVertexBuf, offset: 0, index: 0)

        cmdEncoder.setFragmentBuffer(densityBuf, offset: 0, index: 0)
        var size = size
        cmdEncoder.setFragmentBytes(&size, length: MemoryLayout.size(ofValue: size), index: 1)
        let gridPixelCount = pow(Float(size), 2)
        var maxDensityF = Float(maxDensity)
        var totalDensityScaled = Float(totalDensity) / gridPixelCount * 42
        cmdEncoder.setFragmentBytes(&maxDensityF, length: MemoryLayout.size(ofValue: maxDensityF), index: 2)
        cmdEncoder.setFragmentBytes(&totalDensityScaled, length: MemoryLayout.size(ofValue: totalDensityScaled), index: 3)
        cmdEncoder.setFragmentBytes(&colorScheme, length: MemoryLayout.size(ofValue: colorScheme), index: 4)

        cmdEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        cmdEncoder.endEncoding()

        if let copyback = copyback {
            let copybackEncoder = cmdBuffer.makeBlitCommandEncoder()!
            copybackEncoder.synchronize(resource: copyback)
            copybackEncoder.endEncoding()
        }

        cmdBuffer.commit()
        if awaitCompletion {
            cmdBuffer.waitUntilCompleted()
        }
        return DensityStats(maxDensity: maxDensity, totalDensity: totalDensity)
    }

    func renderAnimation(
        startTime: TimeInterval = 0,
        duration: TimeInterval,
        speed: Double,
        apogeeSlowdown: Double = 1,
        frameRate: Int,
        pointsPerFrame: Int,
        ΔrotationPerSecond: Double = 0.1 * (1 + sqrt(5)) / 2,
        ΔthetaOffsetPerSecond: Double = 0.1
    ) async {
        var assertionID: IOPMAssertionID = 0
        IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            "Rendering fractal" as CFString,
            &assertionID
        )
        defer {
            IOPMAssertionRelease(assertionID)
        }

        let encoder = try! VideoEncoder(width: size, height: size, frameRate: frameRate)
        print("Generating video at: \(encoder.output.path)")

        var timeRendered: Double = 0
        let ΔtBase = speed / Double(frameRate)

        thetaOffset += ΔthetaOffsetPerSecond * speed * startTime
        rotation += ΔrotationPerSecond * speed * startTime

        while timeRendered < duration {
            print("Rendering \(timeRendered) / \(duration) sec",
                "(\(Int(timeRendered / duration * 100))%)",
                "t=\(startTime + timeRendered))...")
            let timer = ContinuousClock.now

            lastCompletedDensityBuf = densityBuf
            densityBuf = gpu.makeBuffer(length: size * size * MemoryLayout<DensityCount>.stride, options: .storageModePrivate)!

            var pointsToRender = pointsPerFrame
            while pointsToRender > 0 {
                pointsToRender -= renderOrbit(maxPoints: pointsToRender)
                await Task.yield()  // In case anybody's waiting to update completed image
            }

            if logTiming {
                print("Rendered frame in \(ContinuousClock.now - timer)")
                print()
            }

            // Show on screen
            Task {
                await MainActor.run(body: self.frameRenderCallback)
            }

            let stats = await self.writeVideoFrame(to: encoder)

            let Δt = ΔtBase / (1 + apogeeSlowdown * (1 * pow((1 - cos(rotation)) / 2, 80)))

            if logStats {
                print("           totalDensity: \(stats.totalDensity)")
                print("             maxDensity: \(stats.maxDensity)")
                print("          concentration: \(stats.concentration)")
            }

            colorScheme.cool = simdColor(h: coolHue.next(speed: Δt), s: coolSat.next(speed: Δt), b: 0.6)
            colorScheme.medium = simdColor(r: medR.next(speed: Δt), g: medG.next(speed: Δt), b: medB.next(speed: Δt))
            colorScheme.hot = simdColor(h: hotHue.next(speed: Δt), s: hotSat.next(speed: Δt), b: 0.9) * 2 - 1

            thetaOffset += ΔthetaOffsetPerSecond * Δt
            rotation += ΔrotationPerSecond * Δt

            timeRendered += Δt / speed
        }

        await encoder.end()
        print("Rendering complete.")
        print(encoder.output.path)
        try! print(FileManager.default.attributesOfItem(atPath: encoder.output.path)[.size] ?? "???", "bytes")
    }

    private func writeVideoFrame(to encoder: VideoEncoder) async -> DensityStats {
        // Many portions adapted from https://github.com/warrenm/MetalOfflineRecording/blob/b4ebaddc37950fd5d835ed60530e7f1905e6d293/MetalOfflineRecording/Renderer.swift
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: size,
            height: size,
            mipmapped: false)
        textureDescriptor.usage = [.renderTarget]
        textureDescriptor.storageMode = .managed
        let outputTexture = gpu.makeTexture(descriptor: textureDescriptor)!
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = outputTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.5, 0.5, 1)

        let stats = await draw(descriptor: renderPassDescriptor, copyback: outputTexture, awaitCompletion: true)
        try! await encoder.writeFrame(forTexture: outputTexture)
        return stats
    }

    func onFrameRendered(action: @escaping @MainActor @Sendable () -> Void) {
        frameRenderCallback = action
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

fileprivate struct Wave {
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

