//
//  ContentView.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import SwiftUI

struct ContentView: View {
    let renderer = MetalFractalRenderer(
        size: 2048,
        rotation: 1.6620565029879701,
        thetaOffset: 3.5144442745012823)

    var timerEnabled = true
    @State var fractalImage: CGImage?

    init(timerEnabled: Bool = true) {
        self.timerEnabled = timerEnabled

        let renderer = renderer
        Task.detached(priority: .medium) {
            await renderer.renderAnimation(frameCount: 600, pointsPerFrame: 1_000_000_000, Δt: 1 / 20.0)
        }
    }

    var body: some View {
        ScrollView([.horizontal, .vertical]) {
            MetalFractalView(renderer: renderer)
        }
    }
}

#Preview {
    return ContentView(timerEnabled: false)
}


extension Sequence {
    func reduceIntoFirst(
        _ updateAccumulatingResult: (inout Element, Self.Element) throws -> ()
    ) rethrows -> Element? {
        var iter = makeIterator()
        guard var result = iter.next() else {
            return nil
        }
        for elem in dropFirst() {
            try updateAccumulatingResult(&result, elem)
        }
        return result
    }
}

