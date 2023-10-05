//
//  ContentView.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import SwiftUI

let metalGrid = MetalGrid(FractalParams(
    rotation: 1.6620565029879701,
    thetaOffset: 3.5144442745012823,
    pointBatchSize: 10000,
    gridSize: 2048,
    randSeed: 0))  // set afresh for each render
let gpuThreadCount = 10000

struct ContentView: View {

    var timerEnabled = true
    @State var fractalImage: CGImage?
    @State var updateFlag = false

    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack {
            let _ = updateFlag
            MetalFractalView(metalGrid: metalGrid, updateFlag: $updateFlag)
                .onReceive(timer) { time in
                    if !timerEnabled {
                        timer.upstream.connect().cancel()
                    }

                    Task {
                        updateFlag.toggle()
                    }
                }
        }
        .padding()
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

