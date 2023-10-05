//
//  PointCounter.metal
//  Density Fractals
//
//  Created by Paul on 9/30/23.
//

#include <metal_stdlib>
#include "Density Fractals-Bridging-Header.h"

using namespace metal;

kernel void renderOrbit(
    constant FractalParams* params,
    device uint* density,
    uint index [[thread_position_in_grid]]
) {
    float sizef = params->gridSize;
    float2x2 rotation = {
        { cos(params->rotation), sin(params->rotation) },
        { -sin(params->rotation), cos(params->rotation) }
    };

    float2 point = {0, 0.5};

    uint64_t rand = params->randSeed ^ index;
    uint64_t randBits = 0;
    int randBitCount = 0;

    for (int n = 0; n < params->pointBatchSize; n++) {
        if (randBitCount <= 0) {
            rand += 0x9e3779b97f4a7c15;
            uint64_t z = rand;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            randBits = z ^ (z >> 31);
            randBitCount = 64;
        }

        if ((randBits & 1) == 0) {
            point *= rotation;
        } else {
            float r = point.x * 0.5 + 0.5;
            float theta = point.y * M_PI_F + params->thetaOffset;
            point = {
                r * cos(theta),
                r * sin(theta)
            };
        }

        randBits >>= 1;
        randBitCount--;

        density[
            int((point.x / 2 + 0.5) * sizef) +
            int((point.y / 2 + 0.5) * sizef) * params->gridSize
        ]++;
    }
}

struct ChunkRange {
    int start, end;
};

ChunkRange computeChunkRange(
    constant FractalParams* params,
    int chunkSize,
    uint chunkIndex
) {
    int start = chunkIndex * chunkSize;
    int end = min(start + chunkSize, params->gridSize * params->gridSize);
    return { start, end };
}

kernel void maxDensity(
    constant FractalParams* params,
    device uint* density,
    device uint* result,
    constant int& chunkSize,
    uint chunkIndex [[thread_position_in_grid]]
) {
    ChunkRange range = computeChunkRange(params, chunkSize, chunkIndex);

    uint max = 0;
    for (int n = range.start; n < range.end; n++) {
        if (density[n] > max) {
            max = density[n];
        }
    }

    result[chunkIndex] = max;
}

kernel void totalDensity(
    constant FractalParams* params,
    device uint* density,
    device uint64_t* result,
    constant int& chunkSize,
    uint chunkIndex [[thread_position_in_grid]]
) {
    ChunkRange range = computeChunkRange(params, chunkSize, chunkIndex);

    uint64_t total = 0;
    for (int n = range.start; n < range.end; n++) {
        total += density[n];
    }

    result[chunkIndex] = total;
}

struct RasterizerData {
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
    constant int& densitySize,
    constant float& maxDensity
) {
    int2 densityPosInt = int2(rint(in.densityPosition * vector_float2(densitySize - 1)));
    uint densityValue = density[densityPosInt.x + densityPosInt.y * densitySize];
    return float4(
        pow(densityValue / maxDensity, 0.3),
        pow(densityValue / maxDensity, 0.5) * 1.5 - 0.5,
        pow(densityValue / maxDensity, 0.7) * 2 - 1,
        1);
}
