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

//private struct RenderVertex {
//    var inPosition: SIMD2<Int>
//    var outPosition: SIMD2<Float>
//}

actor MetalGrid {
    let size = 2048

    private var j = 0

    private let orbit: any MTLBuffer
    private let density: any MTLBuffer
    private let cmdQueue: any MTLCommandQueue
    private let addPointsPipelineState: any MTLComputePipelineState
    private let renderSquarePipelineState: any MTLRenderPipelineState
    private let squareVertexBuf, densitySizeBuf: any MTLBuffer

private var pointCount = 0
private var timer = ContinuousClock.now

    @Published var updateCount = 0

    init() {
        let gpu = MTLCreateSystemDefaultDevice()!
        let gpuLibrary = gpu.makeDefaultLibrary()!

        addPointsPipelineState = try! gpu.makeComputePipelineState(function:
            gpuLibrary.makeFunction(name: "addPoints")!)

        let renderSquareDescriptor = MTLRenderPipelineDescriptor()
        renderSquareDescriptor.vertexFunction = gpuLibrary.makeFunction(name: "densityVertex")!
        renderSquareDescriptor.fragmentFunction = gpuLibrary.makeFunction(name: "densityFragment")!
        renderSquareDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormat.bgra8Unorm
        renderSquarePipelineState = try! gpu.makeRenderPipelineState(descriptor: renderSquareDescriptor)

        orbit = gpu.makeBuffer(length: pointBatchSize * MemoryLayout<PointIndex>.stride, options: .cpuCacheModeWriteCombined)!
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

        let sizeArray = [size, size]
        densitySizeBuf = gpu.makeBuffer(
            bytes: sizeArray,
            length: MemoryLayout.size(ofValue: sizeArray),
            options: .cpuCacheModeWriteCombined)!
    }

    func touchAll(_ points: [(x: Int, y: Int)]) {
        precondition(points.count <= pointBatchSize, "points.count \(points.count) != pointBatchSize \(pointBatchSize)")

        orbit.contents().withMemoryRebound(to: PointIndex.self, capacity: pointBatchSize) { orbits in
            for i in points.indices {
                let point = points[i]
                orbits[i] = point.x + point.y * size
            }
        }

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(addPointsPipelineState)
        cmdEncoder.setBuffer(orbit, offset: 0, index: 0)
        cmdEncoder.setBuffer(density, offset: 0, index: 1)
        cmdEncoder.dispatchThreads(
            MTLSize(width: pointBatchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pointBatchSize, addPointsPipelineState.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()

        pointCount += points.count
    }

    func draw(in drawable: CAMetalDrawable, descriptor: MTLRenderPassDescriptor) {
        print(Double(pointCount) / (ContinuousClock.now - timer).milliseconds, "orbits per ms")
        defer {
            pointCount = 0
            timer = ContinuousClock.now
            print("Done rendering.")
        }

        let unitInterval = 0.0...1.0
        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(.random(in: unitInterval), .random(in: unitInterval), .random(in: unitInterval), 1)
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store

        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        let cmdEncoder = cmdBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        cmdEncoder.setRenderPipelineState(renderSquarePipelineState)

        cmdEncoder.setVertexBuffer(squareVertexBuf, offset: 0, index: 0)
        var sizeBuf = size

        cmdEncoder.setFragmentBuffer(density, offset: 0, index: 0)
        cmdEncoder.setFragmentBytes(&sizeBuf, length: MemoryLayout.size(ofValue: sizeBuf), index: 1)

        cmdEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        cmdEncoder.endEncoding()
        cmdBuffer.present(drawable)
        cmdBuffer.commit()
    }
}
