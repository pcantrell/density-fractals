//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

#include <simd/simd.h>

struct FractalShaderParams {
    float rotation;
    float thetaOffset;

    int gridSize;

    int pointBatchPerThread;
    int gpuThreadCount;

    unsigned long randSeed;

    int chunk, chunkCount;
};

struct FractalColorScheme {
    simd_float3 cool, medium, hot;
};

