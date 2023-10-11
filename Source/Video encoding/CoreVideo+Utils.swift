//
//  CoreVideo+Utils.swift
//  Density Fractals
//
//  Created by Paul on 10/10/23.
//

import CoreVideo

func CVCheckError(_ status: CVReturn, function: StaticString = #function, file: StaticString = #file, line: UInt = #line) {
    if status != kCVReturnSuccess {
        fatalError(
            """
            Core video error: \(CVStatusToString(status))
                in \(function)
                at \(file):\(line)
            """
        )
    }
}

func CVStatusToString(_ status: CVReturn) -> String {
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

