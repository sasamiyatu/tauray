#include "spatial_reprojection_stage.hh"
#include "misc.hh"
#include "camera.hh"
#include <algorithm>

namespace
{
using namespace tr;

shader_source load_source(const tr::spatial_reprojection_stage::options&)
{
    std::map<std::string, std::string> defines;
    return {"shader/spatial_reprojection.comp", defines};
}

struct camera_data_buffer
{
    pmat4 view_proj;
};

struct push_constant_buffer
{
    pvec4 default_value;
    pivec2 size;
    uint32_t source_count;
};

static_assert(sizeof(push_constant_buffer) <= 128);

}

namespace tr
{

spatial_reprojection_stage::spatial_reprojection_stage(
    device_data& dev,
    gbuffer_target& target,
    const options& opt
):  stage(dev),
    current_scene(nullptr),
    target_viewport(target),
    comp(dev, compute_pipeline::params{
        load_source(opt), {}
    }),
    opt(opt),
    camera_data(
        dev,
        sizeof(camera_data_buffer) * opt.active_viewport_count,
        vk::BufferUsageFlagBits::eStorageBuffer
    ),
    stage_timer(
        dev,
        "spatial reprojection (from " +
        std::to_string(opt.active_viewport_count) + " to " +
        std::to_string(target.get_layer_count() - opt.active_viewport_count) +
        " viewports)"
    )
{
    target_viewport.set_layout(vk::ImageLayout::eGeneral);
    this->target_viewport.color.set_layout(vk::ImageLayout::eUndefined);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        comp.update_descriptor_set({
            {"camera_data", {*camera_data, 0, VK_WHOLE_SIZE}},
            {"color_tex", {{}, target_viewport.color[i].view, vk::ImageLayout::eGeneral}},
            {"normal_tex", {{}, target_viewport.normal[i].view, vk::ImageLayout::eGeneral}},
            {"position_tex", {{}, target_viewport.pos[i].view, vk::ImageLayout::eGeneral}}
        }, i);
    }
}

void spatial_reprojection_stage::set_scene(scene* s)
{
    current_scene = s;
    clear_commands();
    if(!current_scene) return;

    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vk::CommandBuffer cb = begin_compute();

        stage_timer.begin(cb, i);

        target_viewport.color.transition_layout_temporary(
            cb, i, vk::ImageLayout::eGeneral, true
        );
        camera_data.upload(i, cb);

        comp.bind(cb, i);

        push_constant_buffer control;
        control.size = target_viewport.get_size();
        control.source_count = opt.active_viewport_count;
        control.default_value = vec4(NAN);

        uvec2 wg = (uvec2(control.size) + 15u)/16u;

        comp.push_constants(cb, control);
        cb.dispatch(wg.x, wg.y, target_viewport.get_layer_count() - control.source_count);

        stage_timer.end(cb, i);
        end_compute(cb, i);
    }
}

void spatial_reprojection_stage::update(uint32_t frame_index)
{
    if(!current_scene) return;

    camera_data.foreach<camera_data_buffer>(
        frame_index,
        opt.active_viewport_count,
        [&](camera_data_buffer& data, size_t i){
            data.view_proj = current_scene->get_camera(i)->get_view_projection();
        }
    );
}

}

