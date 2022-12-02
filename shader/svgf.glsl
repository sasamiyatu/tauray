#ifndef SVGF_GLSL
#define SVGF_GLSL

const float gaussian_kernel[2][2] = {
    { 1.0 / 4.0, 1.0 / 8.0  },
    { 1.0 / 8.0, 1.0 / 16.0 }
};
layout(push_constant) uniform push_constant_buffer
{
    ivec2 size;
    int level;
    int iteration_count;
    int spec_iteration_count;
    int atrous_kernel_radius;
    float sigma_n;
    float sigma_z;
    float sigma_l;
    float temporal_alpha_color;
    float temporal_alpha_moments;
} control;

uint SmallestPowerOf2GreaterThan(in uint x)
{
    // Set all the bits behind the most significant non-zero bit in x to 1.
    // Essentially giving us the largest value that is smaller than the
    // next power of 2 we're looking for.
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    // Return the next power of two value.
    return x + 1;
}

// Returns float precision for a given float value.
// Values within (value -precision, value + precision) map to the same value.
// Precision = exponentRange/MaxMantissaValue = (2^e+1 - 2^e) / (2^NumMantissaBits)
// Ref: https://blog.demofox.org/2017/11/21/floating-point-precision/
float FloatPrecision(in float x, in uint NumMantissaBits)
{
    // Find the exponent range the value is in.
    uint nextPowerOfTwo = SmallestPowerOf2GreaterThan(uint(x));
    float exponentRange = nextPowerOfTwo - (nextPowerOfTwo >> 1);

    float MaxMantissaValue = 1 << NumMantissaBits;

    return exponentRange / MaxMantissaValue;
}


#endif