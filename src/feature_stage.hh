#ifndef TAURAY_FEATURE_STAGE_HH
#define TAURAY_FEATURE_STAGE_HH
#include "rt_camera_stage.hh"

namespace tr
{

class scene;
class feature_stage: public rt_camera_stage
{
public:
    enum feature
    {
        ALBEDO = 0,
        WORLD_NORMAL,
        VIEW_NORMAL,
        WORLD_POS,
        VIEW_POS,
        DISTANCE,
        WORLD_MOTION,
        VIEW_MOTION,
        SCREEN_MOTION,
        INSTANCE_ID
    };

    struct options: public rt_camera_stage::options
    {
        feature feat;
        // Missing rays are filled with the default value.
        vec4 default_value = vec4(NAN);
    };

    feature_stage(
        device_data& dev, 
        uvec2 ray_count,
        const gbuffer_target& output_target,
        const options& opt
    );

protected:
    void record_command_buffer_push_constants(
        vk::CommandBuffer cb,
        uint32_t frame_index,
        uint32_t pass_index
    ) override;

private:
    options opt;
};

}

#endif
