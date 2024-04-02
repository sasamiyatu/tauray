#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

struct material
{
    vec4 albedo_factor;
    vec4 metallic_roughness_factor;
    vec4 emission_factor_double_sided;
    float transmittance;
    float ior;
    float normal_factor;
    float pad;
    int albedo_tex_id;
    int metallic_roughness_tex_id;
    int normal_tex_id;
    int emission_tex_id;
};

struct sampled_material
{
    vec4 albedo;
    float metallic;
    float roughness;
    vec3 emission;
    float transmittance;
    float ior_in;
    float ior_out;
    float f0;
    bool double_sided;
    float shadow_terminator_mul;
};

// Stores the strength of each lobe. Initialize to zero before use!
struct bsdf_lobes
{
    float transmission;
    float diffuse;
    float dielectric_reflection;
    float metallic_reflection;
};

vec3 modulate_bsdf(sampled_material mat, bsdf_lobes bsdf)
{
    return mat.albedo.rgb * (bsdf.metallic_reflection + bsdf.transmission + bsdf.diffuse) + bsdf.dielectric_reflection;
}

vec3 modulate_color(sampled_material mat, vec3 diffuse, vec3 reflected)
{
    return mat.albedo.rgb * mix(diffuse, reflected, mat.metallic) + (1-mat.metallic) * reflected;
}

void add_demodulated_color(
    bsdf_lobes primary_bsdf, vec3 light_color,
    inout vec3 diffuse, inout vec3 reflected
){
    diffuse += light_color * (primary_bsdf.diffuse + primary_bsdf.transmission);
    reflected += light_color * (primary_bsdf.dielectric_reflection + primary_bsdf.metallic_reflection);
}

#endif
