//
//  Util.swift
//  Density Fractals
//
//  Created by Paul on 10/5/23.
//

import Foundation

extension Duration {
    var milliseconds: Double {
        self / .milliseconds(1)
    }
}

extension FixedWidthInteger {
    static var fullRange: ClosedRange<Self> {
        (.min)...(.max)
    }
}

extension ClosedRange where Bound: FixedWidthInteger {
    static var fullRange: Self {
        Bound.fullRange
    }
}
