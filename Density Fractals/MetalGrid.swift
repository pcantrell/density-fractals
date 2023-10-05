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

typealias PointIndex = Int
typealias DensityCount = UInt32

struct MetalFractalView: NSViewRepresentable {
    let metalGrid: MetalGrid

    @Binding var updateFlag: Bool

    // From https://blog.canopas.com/how-to-get-started-with-metal-apis-with-uiview-and-swiftui-124643d8209e#67e6
    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.enableSetNeedsDisplay = true
        view.colorspace = .init(name: CGColorSpace.displayP3)
        return view
    }

    func updateNSView(_ uiView: MTKView, context: Context) {
        uiView.needsDisplay = true
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(metalGrid: metalGrid)
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: MTKView, context: Context) -> CGSize? {
        CGSize(width: metalGrid.size / 2, height: metalGrid.size / 2)
    }

    class Coordinator: NSObject, MTKViewDelegate {
        let metalGrid: MetalGrid

        init(metalGrid: MetalGrid) {
            self.metalGrid = metalGrid
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            print("size will change: \(size)")
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
                print("MetalFractalView will draw")
                await metalGrid.draw(in: drawable, descriptor: descriptor)
                print("MetalFractalView did draw")
            }
        }
    }
}

actor MetalGrid {
    let size: Int

    var params: FractalParams

    var colorScheme: FractalColorScheme

    private var
        coolHue = Wave(Δphase:  pow(0.0030, 0.99)),  // exponents makes speeds all mutually irrational
        coolSat = Wave(Δphase:  pow(0.0070, 0.99), phase: -0.25),
        medR    = Wave(Δphase:  pow(0.0014, 0.99), range: 0...0.5),
        medG    = Wave(Δphase: -pow(0.0025, 0.99), range: 0...0.5),
        medB    = Wave(Δphase:  pow(0.0036, 0.99), range: 0...0.5),
        hotHue  = Wave(Δphase: -pow(0.0053, 0.99))

    private let gpu: any MTLDevice
    private var density: any MTLBuffer
    private let cmdQueue: any MTLCommandQueue
    private let renderOrbitPipelineState: any MTLComputePipelineState
    private let maxDensityPipelineState: any MTLComputePipelineState
    private let totalDensityPipelineState: any MTLComputePipelineState
    private let renderSquarePipelineState: any MTLRenderPipelineState
    private let squareVertexBuf: any MTLBuffer

private var pointCount = 0
private var timer = ContinuousClock.now

    @Published var updateCount = 0

    init(_ params: FractalParams) {
        self.params = params

        size = Int(params.gridSize)

        //CMConvertLuvToXYZ()
        self.colorScheme = FractalColorScheme(
            cool:   simd_float3.zero,
            medium: simd_float3.zero,
            hot:    simd_float3.zero)

        let size = Int(params.gridSize)

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
        renderSquareDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormat.bgra8Unorm
        renderSquarePipelineState = try! gpu.makeRenderPipelineState(descriptor: renderSquareDescriptor)

        density = gpu.makeBuffer(length: size * size * MemoryLayout<DensityCount>.stride, options: .storageModePrivate)!

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

        Task.detached(priority: .medium) {
            while true {
                await metalGrid.renderOrbit()
            }
        }
    }

    func makeOrbitBuffer() -> any MTLBuffer {
        return gpu.makeBuffer(length: Int(params.pointBatchSize) * MemoryLayout<PointIndex>.stride, options: .cpuCacheModeWriteCombined)!
    }

    func renderOrbit() {
//        let timer = ContinuousClock.now
//        defer {
//            print("rendered \(params.pointBatchSize) points in \(ContinuousClock.now - timer)")
//        }

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(renderOrbitPipelineState)

        var params = self.params
        params.randSeed = .random(in: .fullRange)
        cmdEncoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 0)

        cmdEncoder.setBuffer(density, offset: 0, index: 1)

        cmdEncoder.dispatchThreads(
            MTLSize(width: gpuThreadCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(gpuThreadCount, renderOrbitPipelineState.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))

        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        pointCount += Int(params.pointBatchSize) * gpuThreadCount
    }

    func computeMaxDensity() -> DensityCount {
        computeChunked(pipelineState: maxDensityPipelineState)
            .max() ?? 0
    }

    func computeTotalDensity() -> UInt64 {
        computeChunked(pipelineState: totalDensityPipelineState)
            .reduce(0, +)
    }

    private func computeChunked<Result>(
        pipelineState: any MTLComputePipelineState
    ) -> [Result] {
//        print("Computing...")
//        let timer = ContinuousClock.now
//        defer {
//            print("Computed in \(ContinuousClock.now - timer)")
//        }

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(pipelineState)

        var params = self.params
        cmdEncoder.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 0)

        cmdEncoder.setBuffer(density, offset: 0, index: 1)

        let chunkCount = 1000
        let resultBuf = gpu.makeBuffer(length: MemoryLayout<Result>.stride * chunkCount)!
        cmdEncoder.setBuffer(resultBuf, offset: 0, index: 2)

        var chunkSize = (Int(params.gridSize * params.gridSize) + chunkCount - 1) / chunkCount
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

let goldenRatio: Float = (1 + sqrt(5)) / 2

    func draw(in drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor) {
        print(Double(pointCount) / (ContinuousClock.now - timer).milliseconds, "orbits per ms")
        defer {
//            pointCount = 0
//            timer = ContinuousClock.now
params.thetaOffset += 0.1
params.rotation += 0.1 * goldenRatio
let size = Int(params.gridSize)
density = gpu.makeBuffer(length: size * size * MemoryLayout<DensityCount>.stride, options: .storageModePrivate)!
        }

        let Δt = 1.0
        colorScheme.cool = simdColor(h: coolHue.next(speed: Δt), s: coolSat.next(speed: Δt), b: 0.6)
        colorScheme.medium = simdColor(r: medR.next(speed: Δt), g: medG.next(speed: Δt), b: medB.next(speed: Δt))
        colorScheme.hot = simdColor(h: hotHue.next(speed: Δt), s: 1, b: 1)

        let maxDensity = computeMaxDensity()
        let totalDensity = computeTotalDensity()
//        print(
//            Double(maxDensity) / Double(totalDensity) * 1_000_000,
//            Double(maxDensity) / Double(totalDensity) * pow(Double(params.gridSize), 2))
//        print("  max density: \(maxDensity)")
//        print("total density: \(computeTotalDensity())")
//        print("   pointCount: \(pointCount)")

        let unitInterval = 0.0...1.0
        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(.random(in: unitInterval), .random(in: unitInterval), .random(in: unitInterval), 1)
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        cmdEncoder.setRenderPipelineState(renderSquarePipelineState)

        cmdEncoder.setVertexBuffer(squareVertexBuf, offset: 0, index: 0)
        var sizeBuf = params.gridSize

        cmdEncoder.setFragmentBuffer(density, offset: 0, index: 0)
        cmdEncoder.setFragmentBytes(&sizeBuf, length: MemoryLayout.size(ofValue: sizeBuf), index: 1)

        let gridPixelCount = pow(Float(params.gridSize), 2)
        var maxDensityF = Float(maxDensity)
        var totalDensityScaled = Float(totalDensity) / gridPixelCount * 42
        cmdEncoder.setFragmentBytes(&maxDensityF, length: MemoryLayout.size(ofValue: maxDensityF), index: 2)
        cmdEncoder.setFragmentBytes(&totalDensityScaled, length: MemoryLayout.size(ofValue: totalDensityScaled), index: 3)

        cmdEncoder.setFragmentBytes(&colorScheme, length: MemoryLayout.size(ofValue: colorScheme), index: 4)

        cmdEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        cmdEncoder.endEncoding()
        cmdBuffer.present(drawable)
        cmdBuffer.commit()
    }
}

extension Duration {
    var milliseconds: Double {
        self / .milliseconds(1)
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

