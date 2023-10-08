//
//  ContentView.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import SwiftUI

private let golden = (1 + sqrt(5)) / 2

struct ContentView: View {
    let renderer = MetalFractalRenderer(
        size: 2048,
        rotation: 0,
        thetaOffset: 3.5144442745012823 - 1.6620565029879701 / golden)

    var timerEnabled = true
    @State var fractalImage: CGImage?

    init(timerEnabled: Bool = true) {
        self.timerEnabled = timerEnabled

        let renderer = renderer
        Task.detached(priority: .medium) {
            await renderer.renderAnimation(
                duration: 240,
                speed: 2,
                apogeeSlowdown: 2,
                frameRate: 30,
                pointsPerFrame: 1_000_000_000,
                ΔrotationPerSecond: 0.1 * golden,
                ΔthetaOffsetPerSecond: 0.1
            )
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

