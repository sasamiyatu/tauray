#ifndef TAURAY_SVGF_STAGE_HH
#define TAURAY_SVGF_STAGE_HH
#include "stage.hh"
#include "gbuffer.hh"
#include "compute_pipeline.hh"
#include "descriptor_set.hh"
#include "timer.hh"
#include "scene_stage.hh"

namespace tr
{

class svgf_stage: public single_device_stage
{
public:
    struct options
    {
        size_t active_viewport_count = 1;
        int atrous_diffuse_iters;
        int atrous_spec_iters;
        int atrous_kernel_radius;
        float sigma_l;
        float sigma_z;
        float sigma_n;
        float temporal_alpha_color;
        float temporal_alpha_moments;
    };

    svgf_stage(
        device& dev,
        scene_stage& ss,
        gbuffer_target& input_features,
        gbuffer_target& prev_features,
        const options& opt
    );
    svgf_stage(const svgf_stage& other) = delete;
    svgf_stage(svgf_stage&& other) = delete;

    void update(uint32_t frame_index);

    void init_resources();
    void record_command_buffers();

private:
    push_descriptor_set atrous_desc;
    compute_pipeline atrous_comp;
    push_descriptor_set temporal_desc;
    compute_pipeline temporal_comp;
    push_descriptor_set estimate_variance_desc;
    compute_pipeline estimate_variance_comp;
    push_descriptor_set firefly_suppression_desc;
    compute_pipeline firefly_suppression_comp;
    push_descriptor_set disocclusion_fix_desc;
    compute_pipeline disocclusion_fix_comp;
    push_descriptor_set prefilter_variance_desc;
    compute_pipeline prefilter_variance_comp;    
    push_descriptor_set preblur_desc;
    compute_pipeline preblur_comp;    
    push_descriptor_set demodulate_inputs_desc;
    compute_pipeline demodulate_inputs_comp;
    options opt;
    gbuffer_target input_features;
    gbuffer_target prev_features;
    render_target atrous_diffuse_pingpong[2];
    render_target atrous_specular_pingpong[2];
    render_target moments_history[2];
    render_target svgf_color_hist;
    render_target svgf_spec_hist;
    render_target specular_hit_distance_history;
    render_target accumulated_specular_hit_distance;
    render_target emissive; // Needed to store emissive for reconstruction later, since diffuse and specular don't have it and path tracer doesn't store it separately.
    static constexpr uint32_t render_target_count = 11;
    std::unique_ptr<texture> render_target_texture[render_target_count];
    timer svgf_timer;

    sampler my_sampler;

    std::vector<vec4> jitter_history;
    gpu_buffer jitter_buffer;
    scene_stage* ss;
    uint32_t scene_state_counter;

    gpu_buffer uniforms;
};
}

#endif // TAURAY_SVGF_STAGE_HH
