//
//  VideoEncoder.swift
//  Density Fractals
//
//  Created by Paul on 10/6/23.
//

import Foundation
import AVFoundation
import AppKit

// Adapted from:
// - https://stackoverflow.com/a/52799689/239816
// - https://github.com/warrenm/MetalOfflineRecording/blob/master/MetalOfflineRecording/Renderer.swift
//
actor VideoEncoder {
    let output: URL
    let frameRate: Int

    private var assetWriter: AVAssetWriter
    private var assetWriterVideoInput: AVAssetWriterInput
    private var assetWriterPixelBufferInput: AVAssetWriterInputPixelBufferAdaptor

    private(set) var frameNumber: CMTimeValue = 0
    private var completed = false

    init(width: Int, height: Int, frameRate: Int) throws {
        let outputURL = URL.temporaryDirectory.appendingPathComponent("fractal-" + UUID().uuidString + ".mov")
        try self.init(output: outputURL, width: width, height: height, frameRate: frameRate)
    }

    init(output: URL, width: Int, height: Int, frameRate: Int) throws {
        self.output = output
        self.frameRate = frameRate

        assetWriter = try AVAssetWriter(outputURL: output, fileType: .mov)

        assetWriterVideoInput = AVAssetWriterInput(
            mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.proRes422HQ,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoColorPropertiesKey: [  // https://docs.huihoo.com/apple/wwdc/2016/503_advances_in_avfoundation_playback.pdf
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_P3_D65,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
                ]
            ]
        )
        assetWriterVideoInput.expectsMediaDataInRealTime = false
        
        assetWriterPixelBufferInput = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: assetWriterVideoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )
        assetWriter.add(assetWriterVideoInput)

        guard assetWriter.startWriting() else {
            throw VideoEncoderError(message: "assetWriter.startWriting() failed")
        }
        assetWriter.startSession(atSourceTime: .zero)
    }

    func writeFrame(forTexture texture: MTLTexture) async throws {
        precondition(!completed, "VideoEncoder already completed")

        while !assetWriterVideoInput.isReadyForMoreMediaData {
            print("*************** assetWriterVideoInput not ready for more data yet ***************")
            await Task.yield()
        }

        guard let pixelBufferPool = assetWriterPixelBufferInput.pixelBufferPool else {
            throw VideoEncoderError(message: "assetWriterPixelBufferInput has no pixel buffer pool available")
        }
        
        var pixelBuffer: CVPixelBuffer? = nil
        checkCVError(CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer))
        guard let pixelBuffer = pixelBuffer else {
            throw VideoEncoderError(message: "Could not get pixel buffer from asset writer input")
        }
        
        checkCVError(CVPixelBufferLockBaseAddress(pixelBuffer, []))
        let pixelBufferBytes = CVPixelBufferGetBaseAddress(pixelBuffer)!
        
        // Use the bytes per row value from the pixel buffer since its stride may be rounded up to be 16-byte aligned
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        
        texture.getBytes(pixelBufferBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        let presentationTime = CMTime(value: frameNumber, timescale: CMTimeScale(frameRate))
        frameNumber += 1
        guard assetWriterPixelBufferInput.append(pixelBuffer, withPresentationTime: presentationTime) else {
            throw VideoEncoderError(message: "assetWriterPixelBufferInput.append failed")
        }

        checkCVError(CVPixelBufferUnlockBaseAddress(pixelBuffer, []))
    }

    func end() async {
        precondition(!completed, "VideoEncoder already completed")
        completed = true

        assetWriterVideoInput.markAsFinished()
        await assetWriter.finishWriting()
        NSWorkspace.shared.activateFileViewerSelecting([output])
    }
}

struct VideoEncoderError: Error {
    var message: String
}

func checkCVError(_ status: CVReturn, function: StaticString = #function, file: StaticString = #file, line: UInt = #line) {
    if status != kCVReturnSuccess {
        fatalError(
            """
            Core video error: \(cvStatusToString(status))
                in \(function)
                at \(file):\(line)
            """
        )
    }
}

func cvStatusToString(_ status: CVReturn) -> String {
    switch status {
        case kCVReturnSuccess: "kCVReturnSuccess"

        case kCVReturnFirst: "kCVReturnFirst"

        case kCVReturnError: "kCVReturnError"
        case kCVReturnInvalidArgument: "kCVReturnInvalidArgument"
        case kCVReturnAllocationFailed: "kCVReturnAllocationFailed"
        case kCVReturnUnsupported: "kCVReturnUnsupported"

        // DisplayLink related errors
        case kCVReturnInvalidDisplay: "kCVReturnInvalidDisplay"
        case kCVReturnDisplayLinkAlreadyRunning: "kCVReturnDisplayLinkAlreadyRunning"
        case kCVReturnDisplayLinkNotRunning: "kCVReturnDisplayLinkNotRunning"
        case kCVReturnDisplayLinkCallbacksNotSet: "kCVReturnDisplayLinkCallbacksNotSet"

        // Buffer related errors
        case kCVReturnInvalidPixelFormat: "kCVReturnInvalidPixelFormat"
        case kCVReturnInvalidSize: "kCVReturnInvalidSize"
        case kCVReturnInvalidPixelBufferAttributes: "kCVReturnInvalidPixelBufferAttributes"
        case kCVReturnPixelBufferNotOpenGLCompatible: "kCVReturnPixelBufferNotOpenGLCompatible"
        case kCVReturnPixelBufferNotMetalCompatible: "kCVReturnPixelBufferNotMetalCompatible"

        // Buffer Pool related errors
        case kCVReturnWouldExceedAllocationThreshold: "kCVReturnWouldExceedAllocationThreshold"
        case kCVReturnPoolAllocationFailed: "kCVReturnPoolAllocationFailed"
        case kCVReturnInvalidPoolAttributes: "kCVReturnInvalidPoolAttributes"
        case kCVReturnRetry: "kCVReturnRetry"

        case kCVReturnLast: "kCVReturnLast"

        default: "WTF (\(status))"
    }
}

