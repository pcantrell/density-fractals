//
//  PointCounter.metal
//  Density Fractals
//
//  Created by Paul on 9/30/23.
//

#include <metal_stdlib>
#include "Metal-Bridging-Header.h"

using namespace metal;

// MARK: Fractal rendering

/// Splitmix, a very fast pseudorandom number algorithm (https://en.wikipedia.org/wiki/Xorshift)
/// `randState` is the RNG’s ongoing internal state; `result` is the next 64-bit random number.
///
uint64_t nextRand(thread uint64_t& randState) {
    randState += 0x9e3779b97f4a7c15;
    uint64_t z = randState;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

constant int densityBatchCount = 3;
constant int densityBatchCapacity = 10;

void flushDensityBatch(
    device uint* density,
    thread/*group*/ int& densityBatchUsage,
    thread/*group*/ int* densityBatch
) {
    for (int n = densityBatchUsage - 1; n >= 0; n--) {
        density[densityBatch[n]]++;
    }
    densityBatchUsage = 0;
}


/// Repeatedly selects one of two transforms at random, applies that transform to a point in space,
/// and updates the count in the `density` grid as the point moves. The fractal emerges from the
/// distribution of those counts in the density grid.
///
kernel void renderOrbit(
    constant FractalShaderParams& params,
    device uint* density,
    uint threadIndex [[thread_position_in_grid]]
) {
    const float enlargement = 1.1;

    // Precalculate matrix for rotation transform
    float sizef = params.gridSize;
    float2x2 rotation = {
        { cos(params.rotation), sin(params.rotation) },
        { -sin(params.rotation), cos(params.rotation) }
    };

    uint64_t rand = params.randSeed ^ threadIndex;
    uint64_t randBits = 0;
    int randBitCount = 0;

    // Randomly place starting point to provide more even sampling across batches
    // (and fewer artifacts for small point counts)
    randBits = nextRand(rand);
    float initialTheta = randBits % 100000 / 100000.0 * M_PI_F * 2;
    float initialR = randBits / 100000 % 100000 / 200000.0 + 0.3;
    float2 point = {
        initialR * cos(initialTheta),
        initialR * sin(initialTheta)
    };

    int chunkSize = params.gridSize * params.gridSize / params.chunkCount;

    for (int n = 0; n < params.pointBatchPerThread; n++) {
        // Use all 64 bits of randomness from the previous random number before generating another
        if (randBitCount <= 0) {
            randBits = nextRand(rand);
            randBitCount = 64;
        }

        // Randomly select a transform
        if ((randBits & 1) == 0) {
            // Transform 0: rotation
            point *= rotation;
        } else {
            // Transform 1: polar → rectangular coords
            float r = point.x * 0.5 + 0.5;
            float theta = point.y * M_PI_F + params.thetaOffset;
            point = {
                r * cos(theta),
                r * sin(theta)
            };
        }

        randBits >>= 1;
        randBitCount--;

        // Increment appropriate count in density grid
        float2 pixel = (point * enlargement / 2 + 0.5) * sizef;
        if (pixel.x > 0 && pixel.x < sizef && pixel.y > 0 && pixel.y < sizef) {
            int index = int(pixel.x) + int(pixel.y) * params.gridSize;
            if (index / chunkSize == params.chunk) {
                density[index]++;
            }
        }
    }
}

// MARK: Density stats

struct ChunkRange {
    int start, end;
};

/// Helper for max and total density calculations, which both use a map-reduce approach of breaking
/// the computation into chunks, computing the result for each chunk on the GPU, then letting the
/// CPU aggregate all the chunk results.
///
ChunkRange computeChunkRange(
    int gridSize,
    int chunkSize,
    uint chunkIndex
) {
    int start = chunkIndex * chunkSize;
    int end = min(start + chunkSize, gridSize * gridSize);
    return { start, end };
}

/// Computes the max density in a subchunk of the `density` grid.
///
kernel void maxDensity(
    constant int& gridSize,
    device uint* density,
    device uint* result,
    constant int& chunkSize,
    uint chunkIndex [[thread_position_in_grid]]
) {
    ChunkRange range = computeChunkRange(gridSize, chunkSize, chunkIndex);

    uint max = 0;
    for (int n = range.start; n < range.end; n++) {
        if (density[n] > max) {
            max = density[n];
        }
    }

    result[chunkIndex] = max;
}

/// Computes the sum of all densities in a subchunk of the `density` grid.
///
kernel void totalDensity(
    constant int& gridSize,
    device uint* density,
    device uint64_t* result,
    constant int& chunkSize,
    uint chunkIndex [[thread_position_in_grid]]
) {
    ChunkRange range = computeChunkRange(gridSize, chunkSize, chunkIndex);

    uint64_t total = 0;
    for (int n = range.start; n < range.end; n++) {
        total += density[n];
    }

    result[chunkIndex] = total;
}

// MARK: Image rendering & colorization

struct RasterizerData {
    float4 position [[position]];
    float2 densityPosition;
};

vertex RasterizerData densityVertex(
    uint vertexID [[vertex_id]],
    constant float2 *vertices
) {
    RasterizerData out;

    float2 position = vertices[vertexID];

    out.position = vector_float4(0.0, 0.0, 0.0, 1.0);
    out.position.xy = position * 2 - 1;

    out.densityPosition = position;

    return out;
}

/// Converts raw integer values from the `density` grid into pixel colors.
///
fragment float4 densityFragment(
    RasterizerData in [[stage_in]],
    device uint* density,
    constant int& densitySize,
    constant float& maxDensity,
    constant float& totalDensity,
    constant FractalColorScheme& colorScheme
) {
    int2 densityPosInt = int2(rint(in.densityPosition * vector_float2(densitySize - 1)));
    uint densityValue = density[densityPosInt.x + densityPosInt.y * densitySize];

    float scale = max(maxDensity * 0.5, totalDensity);
    float maxWeight = 1 - pow(maxDensity / scale, 0.5);
    float luminance = pow(densityValue / scale, 0.5) + pow(densityValue / maxDensity, 1.5) * maxWeight;
    float hotLuminance = pow(densityValue / maxDensity, 1.5);

    const float medCutoff = 0.12, hotCutoff = 0.3;

    float4 result = 1;
    result.rgb = colorScheme.cool   * pow(luminance, 0.7)
               + colorScheme.medium * pow(max(0.0, (luminance - medCutoff) / (1 - medCutoff)), 1.1)
               + colorScheme.hot    * pow(max(0.0, (hotLuminance - hotCutoff) / (1 - hotCutoff)), 2.5);
    return result;
}

// MARK: Unused color experiments

// from https://stackoverflow.com/a/68542762/239816
float4 rotateHue(
    float4 color,
    float hueAdjust
) {
    const float4  kRGBToYPrime = float4 (0.299, 0.587, 0.114, 0.0);
    const float4  kRGBToI     =  float4 (0.596, -0.275, -0.321, 0.0);
    const float4  kRGBToQ     =  float4 (0.212, -0.523, 0.311, 0.0);
    
    const float4  kYIQToR   =    float4 (1.0, 0.956, 0.621, 0.0);
    const float4  kYIQToG   =    float4 (1.0, -0.272, -0.647, 0.0);
    const float4  kYIQToB   =    float4 (1.0, -1.107, 1.704, 0.0);
    
    // Convert to YIQ
    float   YPrime  = dot (color, kRGBToYPrime);
    float   I      = dot (color, kRGBToI);
    float   Q      = dot (color, kRGBToQ);
    
    // Calculate the hue and chroma
    float   hue     = atan2 (Q, I);
    float   chroma  = sqrt (I * I + Q * Q);
    
    // Make the user's adjustments
    hue += hueAdjust;
    
    // Convert back to YIQ
    Q = chroma * sin(hue);
    I = chroma * cos(hue);
    
    // Convert back to RGB
    float4 yIQ = float4 (YPrime, I, Q, 0.0);
    color.r = dot(yIQ, kYIQToR);
    color.g = dot(yIQ, kYIQToG);
    color.b = dot(yIQ, kYIQToB);
    
    return color;
}

// Adapted from https://web.archive.org/web/20111111074705/http://www.easyrgb.com/index.php?X=MATH&H=17#text17
// and https://web.archive.org/web/20111111080001/http://www.easyrgb.com/index.php?X=MATH&H=01#text1
// with assorted tweaks
float4 luvModified2rgb(float L, float u, float v, float alpha) {
    L *= 100;
    u *= 100;
    v *= 100;

    float uvScale = min(1.0, L / 20);  // prevents oversaturation and discontinuities at low luminances
    u *= uvScale;
    v *= uvScale;

    float var_Y = (L + 16) / 116;
    if (var_Y > 0.206893034422964)
        var_Y = pow(var_Y, 3);
    else
        var_Y = (var_Y - 16 / 116.0) / 7.787;

    const float ref_X =  95.047;   // Observer = 2°, Illuminant = D65
    const float ref_Y = 100.000;
    const float ref_Z = 108.883;
    const float ref_U = (4 * ref_X) / (ref_X + (15 * ref_Y) + (3 * ref_Z));
    const float ref_V = (9 * ref_Y) / (ref_X + (15 * ref_Y) + (3 * ref_Z));

    float var_U = u / (13 * L) + ref_U;
    float var_V = v / (13 * L) + ref_V;

    float Y = var_Y * 100;
    float X =  -(9 * Y * var_U) / ((var_U - 4) * var_V - var_U * var_V);
    float Z = (9 * Y - (15 * var_V * Y) - (var_V * X)) / (3 * var_V);

    float var_X = X / 100;        //X from 0 to  95.047 (Observer = 2°, Illuminant = D65)
    float var_Z = Z / 100;        //Z from 0 to 108.883

    float var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
    float var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
    float var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

    if (var_R > 0.0031308)
        var_R = 1.055 * pow(var_R, 1 / 2.4) - 0.055;
    else
        var_R = 12.92 * var_R;

    if (var_G > 0.0031308)
        var_G = 1.055 * pow(var_G, 1 / 2.4) - 0.055;
    else
        var_G = 12.92 * var_G;

    if (var_B > 0.0031308)
        var_B = 1.055 * pow(var_B, 1 / 2.4) - 0.055;
    else
        var_B = 12.92 * var_B;

    return { var_R, var_G, var_B, alpha };
}

float4 fakeLab2rgb(float luminance, float redGreen, float yellowBlue, float alpha) {
    float r = 1 - redGreen - yellowBlue;
    float g = 1 + redGreen - yellowBlue;
    float b = 1 + yellowBlue - abs(redGreen);
    luminance /= max(r, max(g, b));
    return {
        luminance * r,
        luminance * g,
        luminance * b,
        alpha
    };
}
