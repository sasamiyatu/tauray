#ifndef TAURAY_RENDERER_HH
#define TAURAY_RENDERER_HH
#include "dependency.hh"
#include "scene.hh"

namespace tr
{

class renderer
{
public:
    virtual ~renderer() = default;

    virtual void set_scene(scene* s) = 0;
    virtual void reset_accumulation(bool reset_sample_counter = false) {(void)reset_sample_counter;};
    virtual void render() = 0;
    virtual void set_device_workloads(const std::vector<double>&) {}

private:
};

}

#endif
