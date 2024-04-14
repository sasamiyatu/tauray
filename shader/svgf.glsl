#ifndef SVGF_GLSL
#define SVGF_GLSL

const float gaussian_kernel[2][2] = {
    { 1.0 / 4.0, 1.0 / 8.0  },
    { 1.0 / 8.0, 1.0 / 16.0 }
};

const float gaussian_kernel_5x5[3][3] = {
    {9.0f / 64.0f, 3.0f / 32.0f, 3.0f / 128.0f},
    {3.0f / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f},
    {3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f},
};

layout(push_constant) uniform push_constant_buffer
{
    ivec2 size;
    int iteration;
    int iteration_count;
    int spec_iteration_count;
    int atrous_kernel_radius;
    float sigma_n;
    float sigma_z;
    float sigma_l;
    float temporal_alpha_color;
    float temporal_alpha_moments;
    float min_rect_dim_mul_unproject;
} control;

bool is_in_screen(ivec2 p)
{
    return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, control.size));
}

float saturate(float x) {return clamp(x, 0.0, 1.0);}
vec2 saturate(vec2 x) {return clamp(x, 0.0, 1.0);}
vec3 saturate(vec3 x) {return clamp(x, 0.0, 1.0);}
vec4 saturate(vec4 x) {return clamp(x, 0.0, 1.0);}

vec3 viridis_quintic( float x )
{
	x = saturate( x );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.280268003, -0.143510503, +2.225793877, -14.815088879 ) ) + dot( x2.xy, vec2( +25.212752309, -11.772589584 ) ),
		dot( x1.xyzw, vec4( -0.002117546, +1.617109353, -1.909305070, +2.701152864 ) ) + dot( x2.xy, vec2( -1.685288385, +0.178738871 ) ),
		dot( x1.xyzw, vec4( +0.300805501, +2.614650302, -12.019139090, +28.933559110 ) ) + dot( x2.xy, vec2( -33.491294770, +13.762053843 ) ) );
}

float get_specular_dominant_factor(float n_dot_v, float roughness)
{
    // https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    float l = 0.298475f * log(39.4115f - 39.0029f * roughness);
    return clamp(pow(max(1.0f - n_dot_v, 0.0f), 10.8649f) * (1.0f - l) + l, 0.0f, 1.0f);
}

// View should point away from surface.
vec3 get_specular_dominant_dir(vec3 view, vec3 normal, float roughness)
{
    float n_dot_v = dot(normal, view);
    vec3 refl = reflect(view, normal);
    return normalize(mix(normal, refl, get_specular_dominant_factor(n_dot_v, roughness)));
}

// Derived through isotropic spherical gaussians; may be inaccurate. 1.0f acts
// as diffuse.
// Similarity metric from: "Specular Lobe-Aware Filtering and Upsampling for
// Interactive Indirect Illumination"
float specular_lobe_similarity(
    vec3 lobe_dir1, float ndotv1, float roughness1,
    vec3 lobe_dir2, float ndotv2, float roughness2
){
    // Exact
    //float sharpness1 = 0.533f / (roughness1 * roughness1 * max(ndotv1, 0.0001f));
    //float sharpness2 = 0.533f / (roughness2 * roughness2 * max(ndotv2, 0.0001f));
    //float amplitude1 = sqrt(sharpness1 / (M_PI * (1.0f - exp(-4.0f * sharpness1))));
    //float amplitude2 = sqrt(sharpness2 / (M_PI * (1.0f - exp(-4.0f * sharpness2))));
    //float dm = length(sharpness1 * lobe_dir1 + sharpness2 * lobe_dir2);
    //float expo = exp(dm - sharpness1 - sharpness2) * amplitude1 * amplitude2;
    //float other = 1.0f - exp(-2.0f * dm);
    //return (2.0f * M_PI * expo * other) / dm;
    // Approximate
    float sharpness1 = 0.533f / (roughness1 * roughness1 * max(ndotv1, 0.0001f));
    float sharpness2 = 0.533f / (roughness2 * roughness2 * max(ndotv2, 0.0001f));
    float dm = length(sharpness1 * lobe_dir1 + sharpness2 * lobe_dir2);
    float expo = exp(dm - sharpness1 - sharpness2) * sqrt(sharpness1 * sharpness2);
    return clamp((2.0f * expo) / dm, 0.0f, 1.0f);
}
// View points away from surface.
float specular_lobe_similarity(
    vec3 view1, vec3 normal1, float roughness1,
    vec3 view2, vec3 normal2, float roughness2
){
    vec3 axis1 = get_specular_dominant_dir(view1, normal1, roughness1);
    vec3 axis2 = get_specular_dominant_dir(view2, normal2, roughness2);
    return specular_lobe_similarity(
        axis1, dot(view1, normal1), roughness1,
        axis2, dot(view2, normal2), roughness1
    );
}

float get_specular_lobe_half_angle(float percentage_of_energy, float roughness)
{
    return atan(roughness * sqrt(percentage_of_energy / (1.0 - percentage_of_energy)));
}


//====================================================================================
// Configurable params
//====================================================================================

// Base blur radius in pixels, actually blur radius is scaled by frustum size
#define PREPASS_BLUR_RADIUS 30.0

#define TEMPORAL_ACCUMULATION_USE_BICUBIC_FILTER 1

#define ATROUS_ITERATIONS 5
#define ATROUS_RADIUS 1 

// Randomly offset sample positions to mitigate "ringing"
#define ATROUS_RANDOM_OFFSET 1

//====================================================================================
// Toggle different passes on and off, useful for debugging
//====================================================================================

#define DENOISING_ENABLED 1

#if DENOISING_ENABLED == 1

#define PREPASS_ENABLED 1

#define TEMPORAL_ACCUMULATION_ENABLED 1

#define DISOCCLUSION_FIX_ENABLED 1
#define DISOCCLUSION_FIX_USE_EDGE_STOPPERS 1

#define FIREFLY_SUPPRESSION_ENABLED 1

#define SPATIAL_VARIANCE_ESTIMATE_ENABLED 1

#define PREFILTER_VARIANCE_ENABLED 1

#define ATROUS_ENABLED 1

#endif

#define OUTPUT_DENOISED_DIFFUSE 0
#define OUTPUT_VARIANCE 1
#define OUTPUT_HIST_LENGTH 2
#define OUTPUT_UNFILTERED_VARIANCE 3
#define OUTPUT_REMODULATED_DENOISED_DIFFUSE 4
#define OUTPUT_DENOISED_SPECULAR 5
#define OUTPUT_REMODULATED_DENOISED_DIFFUSE_AND_SPECULAR 6

#define FINAL_OUTPUT 6

#define MAX_ACCUMULATED_FRAMES 32

#endif // SVGF_GLSL