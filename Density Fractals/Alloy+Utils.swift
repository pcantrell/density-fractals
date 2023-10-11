//
//  Alloy+Utils.swift
//  Density Fractals
//
//  Created by Paul on 10/10/23.
//

import Alloy

public extension MTLContext {
    func schedule(wait: Bool, _ bufferEncodings: (MTLCommandBuffer) throws -> Void) throws {
        return wait
            ? try scheduleAndWait(bufferEncodings)
            : try schedule(bufferEncodings)
    }
}
