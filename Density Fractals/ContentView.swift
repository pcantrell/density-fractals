//
//  ContentView.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import SwiftUI

struct ContentView: View {
    var timerEnabled = true
    @State var fractalImage: CGImage?
    @State var mult = 1

    let timer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack {
            if let fractalImage = fractalImage {
                Image(decorative: fractalImage, scale: 2)
            }
        }
        .padding()
        .onReceive(timer) { time in
            if !timerEnabled {
                timer.upstream.connect().cancel()
            }

            for fractal in FractalRenderer.pauls {
                Task.detached { await fractal.orbit() }
            }

            Task {
                var result = await FractalRenderer.pauls.first?.grid

                for renderer in FractalRenderer.pauls.dropFirst() {
                    result!.merge(await renderer.grid)
                }

                fractalImage = result?.renderImage()
                print("Updated image at \(time)")
            }
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

