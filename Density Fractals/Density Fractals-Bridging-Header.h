//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

#include <simd/simd.h>

struct RenderVertex {
    vector_float2 position;
};

struct FractalParams {
    float rotation;
    float thetaOffset;

    int gridSize;

    int pointBatchSize;
    int gpuThreadCount;

    unsigned long randSeed;
};

struct FractalColorScheme {
    simd_float3 cool, medium, hot;
};

