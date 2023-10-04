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
        return view
    }

    func updateNSView(_ uiView: MTKView, context: Context) {
        uiView.needsDisplay = true
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(metalGrid: metalGrid)
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
    let params: FractalParams

    private let gpu: any MTLDevice
    private let density: any MTLBuffer
    private let cmdQueue: any MTLCommandQueue
    private let renderOrbitPipelineState: any MTLComputePipelineState
    private let renderSquarePipelineState: any MTLRenderPipelineState
    private let squareVertexBuf: any MTLBuffer

private var pointCount = 0
private var timer = ContinuousClock.now

    @Published var updateCount = 0

    init(_ params: FractalParams) {
        self.params = params

        let size = Int(params.gridSize)

        gpu = MTLCreateSystemDefaultDevice()!
        let gpuLibrary = gpu.makeDefaultLibrary()!

        renderOrbitPipelineState = try! gpu.makeComputePipelineState(function:
            gpuLibrary.makeFunction(name: "renderOrbit")!)

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
        let timer = ContinuousClock.now
        defer {
            print("rendered \(params.pointBatchSize) points in \(ContinuousClock.now - timer)")
        }

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

    func draw(in drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor) {
        print(Double(pointCount) / (ContinuousClock.now - timer).milliseconds, "orbits per ms")
        defer {
            pointCount = 0
            timer = ContinuousClock.now
        }

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
