//
//  PointCounter.metal
//  Density Fractals
//
//  Created by Paul on 9/30/23.
//

#include <metal_stdlib>
#include "Density Fractals-Bridging-Header.h"

using namespace metal;

kernel void addPoints(
    device const int* orbit,
    device uint* density,
    uint index [[thread_position_in_grid]]
) {
    density[orbit[index]]++;
}

struct RasterizerData
{
    float4 position [[position]];
    float2 densityPosition;
};

vertex RasterizerData densityVertex(
    uint vertexID [[vertex_id]],
    constant RenderVertex *vertices
) {
    RasterizerData out;

    vector_float2 position = vertices[vertexID].position;

    out.position = vector_float4(0.0, 0.0, 0.0, 1.0);
    out.position.xy = position * 2 - 1;

    out.densityPosition = position;

    return out;
}


fragment float4 densityFragment(
    RasterizerData in [[stage_in]],
    device uint* density,
    constant int *densitySize
) {
    int2 densityPosInt = int2(rint(in.densityPosition * vector_float2(*densitySize - 1)));
    uint densityValue = density[densityPosInt.x + densityPosInt.y * *densitySize];
    return float4(pow(densityValue, 0.3) / 10.0, pow(densityValue, 0.5) / 100.0, pow(densityValue, 0.7) / 1000.0, 1);
//    return float4(densityPosInt.x / 2048.0, densityPosInt.y / 2048.0, 1, 1);
}


