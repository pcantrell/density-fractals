//
//  ContentView.swift
//  Density Fractals
//
//  Created by Paul on 9/28/23.
//

import SwiftUI
import UniformTypeIdentifiers
import IOKit.pwr_mgt

struct ContentView: View {
    @State var rotation    = Parameter(name: "rotation",    value:
        1.724643921305295)
    //( 1.6620565029879701
    //+ 1.5247313396226190) / 2)
    @State var thetaOffset = Parameter(name: "thetaOffset", value:
        3.0466792337230033)
    //( 3.5144442745012823
    //+ 3.0289141929447223) / 2)

    @State var renderBatchSize: Float = 2
    @State var renderLoopEnabled = false

    @State var fractalImage: CGImage?
    @State var maxCount: Count?

    @State var curRenderVersion = 0

    @State var sleepPrevention: IOPMAssertionID = 0

    var body: some View {
        VStack {
            HStack {
                Grid {
                    GridRow {
                        Text("\(rotation.name):").gridColumnAlignment(.trailing)
                        Text(verbatim: String(rotation.value)).gridColumnAlignment(.leading)

                        Button("+") {
                            stepParam(&rotation, direction: .increase)
                        }.keyboardShortcut(.upArrow, modifiers: [])
                        Button("-") {
                            stepParam(&rotation, direction: .decrease)
                        }.keyboardShortcut(.downArrow, modifiers: [])

                        Text("±")
                        Text(verbatim: String(rotation.step)).gridColumnAlignment(.leading)

                        Button("+") {
                            changeStep(&rotation, direction: .increase)
                        }.keyboardShortcut(.upArrow, modifiers: [.command])
                        Button("-") {
                            changeStep(&rotation, direction: .decrease)
                        }.keyboardShortcut(.downArrow, modifiers: [.command])
                    }

                    GridRow {
                        Text("\(thetaOffset.name):").gridColumnAlignment(.trailing)
                        Text(verbatim: String(thetaOffset.value)).gridColumnAlignment(.leading)

                        Button("+") {
                            stepParam(&thetaOffset, direction: .increase)
                        }.keyboardShortcut(.rightArrow, modifiers: [])
                        Button("-") {
                            stepParam(&thetaOffset, direction: .decrease)
                        }.keyboardShortcut(.leftArrow, modifiers: [])

                        Text("±")
                        Text(verbatim: String(thetaOffset.step)).gridColumnAlignment(.leading)

                        Button("+") {
                            changeStep(&thetaOffset, direction: .increase)
                        }.keyboardShortcut(.rightArrow, modifiers: [.command])
                        Button("-") {
                            changeStep(&thetaOffset, direction: .decrease)
                        }.keyboardShortcut(.leftArrow, modifiers: [.command])
                    }
                }
                .padding()

                VStack {
                    HStack {
                        Text("Rendering batch size")
                        Slider(value: $renderBatchSize, in: 2...14) { changeInProgress in
                            if !changeInProgress {
                                cancelCurrentRendering()
                                render()
                            }
                        }
                    }

                    HStack(spacing: 20) {
                        Text("Quality: \(maxCount.map(String.init) ?? "-")")
                        Toggle("Keep rendering", isOn: $renderLoopEnabled)
                            .onChange(of: renderLoopEnabled) {
                                if renderLoopEnabled {
                                    preventSleep()
                                    render()
                                } else {
                                    allowSleep()
                                }
                            }
                        Button("Save Image", action: saveImage)
                    }
                }
            }

            ScrollView([.horizontal, .vertical]) {
                if let fractalImage = fractalImage {
                    Image(decorative: fractalImage, scale: 2)
                }
            }
        }
    }

    func stepParam(_ param: inout Parameter, direction: Direction) {
        let factor = switch direction {
            case .increase: 1.0
            case .decrease: -1.0
        }
        param.value += param.step * factor
        clearAndRender()
    }

    func changeStep(_ param: inout Parameter, direction: Direction) {
        let factor = switch direction {
            case .increase: 2.0
            case .decrease: 0.5
        }
        param.step *= factor
    }

    func saveImage() {
        Task {
            let fractalImage = await FractalRenderer.grid.renderImage().image
            let imageURL = URL.temporaryDirectory.appendingPathComponent("fractal-" + UUID().uuidString + ".png")
            let imageDest = CGImageDestinationCreateWithURL(
                imageURL as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(imageDest, fractalImage, nil)
            CGImageDestinationFinalize(imageDest)
            NSWorkspace.shared.activateFileViewerSelecting([imageURL])
        }
    }

    enum Direction {
        case increase
        case decrease
    }

    func cancelCurrentRendering() {
        curRenderVersion += 1
    }

    func clearAndRender() {
        cancelCurrentRendering()
        Task {
            await FractalRenderer.grid.clear()
            render()
        }
    }

    func render(_ versionToRender: Int? = nil) {
        let versionToRender = versionToRender ?? curRenderVersion

        Task {
            guard versionToRender == curRenderVersion else {
                print("Outdated render version")
                return
            }

            await withTaskGroup(of: Void.self) { group in
                for renderer in FractalRenderer.renderers {
                    group.addTask {
                        await renderer.orbit(
                            rotation: rotation.value,
                            thetaOffset: thetaOffset.value,
                            batches: Int(pow(2, renderBatchSize)))
                    }
                }
            }
            (fractalImage, maxCount) = await FractalRenderer.grid.renderImage()

            if renderLoopEnabled {
                render(versionToRender)
            }
        }
    }

    private func preventSleep() {
        if sleepPrevention == 0 {
            IOPMAssertionCreateWithName(
                kIOPMAssertionTypePreventSystemSleep as CFString,
                IOPMAssertionLevel(kIOPMAssertionLevelOn),
                "Rendering fractal" as CFString,
                &sleepPrevention
            )
        }
    }
    
    private func allowSleep() {
        if sleepPrevention != 0 {
            IOPMAssertionRelease(sleepPrevention)
            sleepPrevention = 0
        }
    }

    struct Parameter: Identifiable {
        var name: String
        var value: Double
        var step: Double = 0.1

        var id: String { name }
    }
}


#Preview {
    return ContentView()
}
