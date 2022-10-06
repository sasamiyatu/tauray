#include "whitted_stage.hh"
#include "scene.hh"
#include "environment_map.hh"

namespace
{
using namespace tr;

namespace whitted
{
    shader_sources load_sources(whitted_stage::options opt)
    {
        std::map<std::string, std::string> defines;
        rt_camera_stage::get_common_defines(defines, opt);
        return {
            {}, {},
            {"shader/whitted.rgen", defines},
            {
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/whitted.rchit"},
                    {"shader/whitted.rahit"}
                },
                {
                    vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                    {"shader/transmission_shadow.rchit"},
                    {"shader/transmission_shadow.rahit"}
                }
            },
            {
                {"shader/whitted.rmiss"},
                {"shader/transmission_shadow.rmiss"}
            }
        };
    }

    struct push_constant_buffer
    {
        uint32_t directional_light_count;
        uint32_t point_light_count;
        uint32_t max_depth;
        // -1 for no environment map
        int environment_proj;
        pvec4 environment_factor;
        pvec4 ambient;
        float min_ray_dist;
    };

    // The minimum maximum size for push constant buffers is 128 bytes in vulkan.
    static_assert(sizeof(push_constant_buffer) <= 128);
}

gfx_pipeline::pipeline_state build_state(
    gfx_pipeline::pipeline_state state,
    const whitted_stage::options& opt
){
    state.rt_options.max_recursion_depth = max(opt.max_ray_depth, 1);
    return state;
}

}

namespace tr
{

whitted_stage::whitted_stage(
    device_data& dev,
    uvec2 ray_count,
    const gbuffer_target& output_target,
    const options& opt
):  rt_camera_stage(
        dev, output_target,
        build_state(rt_stage::get_common_state(
            ray_count, uvec4(0,0,output_target.get_size()),
            whitted::load_sources(opt), opt
        ), opt),
        opt
    ),
    opt(opt)
{
}

void whitted_stage::record_command_buffer_push_constants(
    vk::CommandBuffer cb,
    uint32_t /*frame_index*/,
    uint32_t /*pass_index*/
){
    scene* cur_scene = get_scene();
    whitted::push_constant_buffer control;
    control.directional_light_count = cur_scene->get_directional_lights().size();
    control.point_light_count =
        cur_scene->get_point_lights().size() +
        cur_scene->get_spotlights().size();
    control.max_depth = opt.max_ray_depth;

    environment_map* envmap = cur_scene->get_environment_map();
    if(envmap)
    {
        control.environment_factor = vec4(envmap->get_factor(), 1);
        control.environment_proj = (int)envmap->get_projection();
    }
    else
    {
        control.environment_factor = vec4(0);
        control.environment_proj = -1;
    }
    control.ambient = pvec4(cur_scene->get_ambient(), 1);
    control.min_ray_dist = opt.min_ray_dist;

    gfx.push_constants(cb, control);
}

}
