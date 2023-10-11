//
//  MetalFractalView.swift
//  Density Fractals
//
//  Created by Paul on 10/10/23.
//

import MetalKit
import SwiftUI

struct MetalFractalView: NSViewRepresentable {
    let renderer: MetalFractalRenderer

    // From https://blog.canopas.com/how-to-get-started-with-metal-apis-with-uiview-and-swiftui-124643d8209e#67e6
    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.enableSetNeedsDisplay = true
        view.framebufferOnly = false
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
            // We always render to fixed-size texture matching renderer size, not window size
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
                if await metalGrid.drawLastCompleted(descriptor: descriptor) {
                    drawable.present()
                }
            }
        }
    }
}

