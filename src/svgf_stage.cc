#include "svgf_stage.hh"
#include "misc.hh"
#include "camera.hh"

namespace
{
using namespace tr;

struct push_constant_buffer_atrous
{
    pivec2 size;
    int parity;
    int iteration;
    int stride;
    int iteration_count;
};

struct push_constant_buffer_temporal
{
    pivec2 size;
};

static_assert(sizeof(push_constant_buffer_atrous) <= 128);
static_assert(sizeof(push_constant_buffer_temporal) <= 128);
}

namespace tr
{

svgf_stage::svgf_stage(
    device_data& dev,
    gbuffer_target& input_features,
    gbuffer_target& prev_features,
    const options& opt
) : stage(dev),
    atrous_comp(dev, compute_pipeline::params{shader_source("shader/svgf_atrous.comp"), {}}),
    temporal_comp(dev, compute_pipeline::params{shader_source("shader/svgf_temporal.comp"), {}}),
    estimate_variance_comp(dev, compute_pipeline::params{ shader_source("shader/svgf_estimate_variance.comp"), {} }),
    opt(opt),
    input_features(input_features),
    prev_features(prev_features),
    svgf_timer(dev, "svgf (" + std::to_string(input_features.get_layer_count()) + " viewports)"),
    jitter_buffer(dev, sizeof(pvec4)* 1, vk::BufferUsageFlagBits::eStorageBuffer)
{
    init_resources();
    record_command_buffers();
}

void svgf_stage::set_scene(scene* cur_scene)
{
    this->cur_scene = cur_scene;
}

void svgf_stage::update(uint32_t frame_index)
{
    bool existing = jitter_history.size() != 0;
    int viewport_count = 1; // FIX THIS
    jitter_history.resize(viewport_count);


    for (size_t i = 0; i < viewport_count; ++i)
    {
        vec4& v = jitter_history[i];
        vec2 cur_jitter = cur_scene->get_camera(i)->get_jitter();
        vec2 prev_jitter = v;
        if (!existing) prev_jitter = cur_jitter;
        v = vec4(cur_jitter, prev_jitter);
    }

    jitter_buffer.update(frame_index, jitter_history.data());
    //printf("jitter hist: %f %f %f %f\n", jitter_history[0].x, jitter_history[0].y, jitter_history[0].z, jitter_history[0].w);
}

void svgf_stage::init_resources()
{
    for (int i = 0; i < 8; ++i)
    {
        render_target_texture[i].reset(new texture(
            *dev,
            input_features.color.get_size(),
            input_features.get_layer_count(),
            vk::Format::eR16G16B16A16Sfloat,
            0, nullptr,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::ImageLayout::eGeneral,
            vk::SampleCountFlagBits::e1
        ));
    }

    int rt_index = 0;
    atrous_specular_pingpong[0] = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    atrous_specular_pingpong[1] = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    moments_history[0]          = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    moments_history[1]          = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    svgf_color_hist             = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    svgf_spec_hist              = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    atrous_diffuse_pingpong[0]  = render_target_texture[rt_index++]->get_array_render_target(dev->index);
    atrous_diffuse_pingpong[1]  = render_target_texture[rt_index++]->get_array_render_target(dev->index);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        atrous_comp.update_descriptor_set({
            {"color_ping", {{}, atrous_diffuse_pingpong[1][i].view, vk::ImageLayout::eGeneral} },
            {"color_pong", {{}, atrous_diffuse_pingpong[0][i].view, vk::ImageLayout::eGeneral} },
            {"specular_ping", {{}, atrous_specular_pingpong[1][i].view, vk::ImageLayout::eGeneral} },
            {"specular_pong", {{}, atrous_specular_pingpong[0][i].view, vk::ImageLayout::eGeneral} },
            {"final_output", {{}, input_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"color_hist", {{}, svgf_color_hist[i].view, vk::ImageLayout::eGeneral}},
            {"spec_hist", {{}, svgf_spec_hist[i].view, vk::ImageLayout::eGeneral}},
            {"in_linear_depth", {{}, input_features.linear_depth[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo[i].view, vk::ImageLayout::eGeneral}},
        }, i);
        temporal_comp.update_descriptor_set({
            {"in_color", {{}, input_features.color[i].view, vk::ImageLayout::eGeneral}},
            {"in_diffuse", {{}, input_features.diffuse[i].view, vk::ImageLayout::eGeneral}},
            {"previous_color", {{}, svgf_color_hist[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_screen_motion", {{}, input_features.screen_motion[i].view, vk::ImageLayout::eGeneral}},
            {"previous_normal", {{}, prev_features.normal[i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo[i].view, vk::ImageLayout::eGeneral}},
            {"previous_moments", {{}, moments_history[0][i].view, vk::ImageLayout::eGeneral}},
            {"out_moments", {{}, moments_history[1][i].view, vk::ImageLayout::eGeneral}},
            {"out_color", {{}, atrous_diffuse_pingpong[0][i].view, vk::ImageLayout::eGeneral}},
            {"out_specular", {{}, atrous_specular_pingpong[0][i].view, vk::ImageLayout::eGeneral} },
            {"in_linear_depth", {{}, input_features.linear_depth[i].view, vk::ImageLayout::eGeneral}},
            {"previous_linear_depth", {{}, prev_features.linear_depth[i].view, vk::ImageLayout::eGeneral}},
            {"jitter_info", {*jitter_buffer, 0, VK_WHOLE_SIZE}},
            {"previous_specular", {{}, svgf_spec_hist[i].view, vk::ImageLayout::eGeneral}},
        }, i);
        estimate_variance_comp.update_descriptor_set({
            {"in_color", {{}, atrous_diffuse_pingpong[0][i].view, vk::ImageLayout::eGeneral}},
            {"out_color", {{}, atrous_diffuse_pingpong[1][i].view, vk::ImageLayout::eGeneral}},
            {"in_specular", {{}, atrous_specular_pingpong[0][i].view, vk::ImageLayout::eGeneral} },
            {"out_specular", {{}, atrous_specular_pingpong[1][i].view, vk::ImageLayout::eGeneral} },
            {"in_linear_depth", {{}, input_features.linear_depth[i].view, vk::ImageLayout::eGeneral}},
            {"color_hist", {{}, svgf_color_hist[i].view, vk::ImageLayout::eGeneral}},
            {"current_moments", {{}, moments_history[1][i].view, vk::ImageLayout::eGeneral}},
            {"moments_hist", {{}, moments_history[0][i].view, vk::ImageLayout::eGeneral}},
            {"in_albedo", {{}, input_features.albedo[i].view, vk::ImageLayout::eGeneral}},
            {"in_normal", {{}, input_features.normal[i].view, vk::ImageLayout::eGeneral}},
        }, i);
    }
}

void svgf_stage::record_command_buffers()
{
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        svgf_timer.begin(cb, i);

        jitter_buffer.upload(i, cb);
        
        uvec2 wg = (input_features.get_size()+15u)/16u;
        push_constant_buffer_temporal control_temporal;
        control_temporal.size = input_features.get_size();
        temporal_comp.bind(cb, i);
        temporal_comp.push_constants(cb, control_temporal);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead
        };
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        estimate_variance_comp.bind(cb, i);
        estimate_variance_comp.push_constants(cb, control_temporal);
        cb.dispatch(wg.x, wg.y, input_features.get_layer_count());

        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, barrier, {}, {}
        );

        atrous_comp.bind(cb, i);
        const int iteration_count = 4;
        for (int j = 0; j < iteration_count; ++j)
        {
            if (j != 0)
            {
                cb.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {}, barrier, {}, {}
                );
            }
            push_constant_buffer_atrous control_atrous;
            control_atrous.size = input_features.get_size();
            control_atrous.iteration_count = iteration_count;
            control_atrous.iteration = j;
            atrous_comp.push_constants(cb, control_atrous);
            cb.dispatch(wg.x, wg.y, input_features.get_layer_count());
        }

        //cb.pipelineBarrier(
        //    vk::PipelineStageFlagBits::eComputeShader,
        //    vk::PipelineStageFlagBits::eComputeShader,
        //    {}, barrier, {}, {}
        //);

#if 0
        push_constant_buffer_atrous control_atrous;
        control_atrous.size = input_features.get_size();
        control_atrous.iteration_count = opt.repeat_count;

        for (int j = 0; j < opt.repeat_count; ++j)
        {
            if (j != 0)
            {
                vk::MemoryBarrier barrier{
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead
                };
                cb.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {}, barrier, {}, {}
                );
            }
            control_atrous.parity = (j-1) % 2;
            control_atrous.iteration = j;
            control_atrous.stride = j;//opt.repeat_count-1-j;
            atrous_comp.bind(cb, i);
            atrous_comp.push_constants(cb, control_atrous);
            cb.dispatch(wg.x, wg.y, input_features.get_layer_count());
        }

#endif
        svgf_timer.end(cb, i);
        end_compute(cb, i);
    }
}

}
