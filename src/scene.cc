#include "scene.hh"
#include "camera.hh"
#include "misc.hh"
#include "shadow_map_renderer.hh"
#include "placeholders.hh"
#include "environment_map.hh"

namespace tr
{

scene::scene(
    context& ctx,
    size_t max_instances,
    size_t max_lights
):  light_scene(ctx, max_lights),
    mesh_scene(ctx, max_instances),
    ctx(&ctx),
    total_ticks(0),
    smr(nullptr),
    sh_grid_textures(nullptr)
{
    std::vector<device_data>& devices = ctx.get_devices();
    for(size_t i = 0; i < devices.size(); ++i)
        scene_buffers.emplace_back(devices[i]);

    init_acceleration_structures();
}

void scene::set_camera(camera* cam)
{
    cameras = {cam};
}

camera* scene::get_camera(unsigned index) const
{
    if(index >= cameras.size()) return nullptr;
    return cameras[index];
}

void scene::add(camera& c) { unsorted_insert(cameras, &c); }
void scene::remove(camera& c) { unsorted_erase(cameras, &c); }
void scene::clear_cameras()
{
    cameras.clear();
}

const std::vector<camera*>& scene::get_cameras() const
{
    return cameras;
}

void scene::reorder_cameras_by_active(const std::set<int>& active_indices)
{
    std::vector<camera*> new_cameras;
    for(size_t i = 0; i < cameras.size(); ++i)
    {
        if(active_indices.count(i))
            new_cameras.push_back(cameras[i]);
    }
    for(size_t i = 0; i < cameras.size(); ++i)
    {
        if(!active_indices.count(i))
            new_cameras.push_back(cameras[i]);
    }
    cameras = std::move(new_cameras);
}

void scene::set_camera_jitter(const std::vector<vec2>& jitter)
{
    for(camera* cam: cameras)
        cam->set_jitter(jitter);
}

void scene::add_control_node(animated_node& o)
{ sorted_insert(control_nodes, &o); }
void scene::remove_control_node(animated_node& o)
{ sorted_erase(control_nodes, &o); }
void scene::clear_control_nodes() { control_nodes.clear(); }

void scene::clear()
{
    clear_cameras();
    clear_mesh_objects();
    clear_point_lights();
    clear_spotlights();
    clear_directional_lights();
    clear_control_nodes();
}

void scene::play(const std::string& name, bool loop, bool use_fallback)
{
    auto play_handler = [&](animated_node* n){
        n->play(name, loop, use_fallback);
    };
    for(camera* c: cameras) play_handler(c);
    for(animated_node* o: control_nodes) play_handler(o);

    light_scene::visit_animated(play_handler);
    mesh_scene::visit_animated(play_handler);
}

void scene::update(time_ticks dt, bool force_update)
{
    for(camera* c: cameras)
    {
        c->step_jitter();
    }

    if(dt > 0 || force_update)
    {
        auto update_handler = [&](animated_node* n){
            n->update(dt);
        };
        for(camera* c: cameras)
            update_handler(c);
        for(animated_node* o: control_nodes) update_handler(o);
        light_scene::visit_animated(update_handler);
        mesh_scene::visit_animated(update_handler);
    }
    total_ticks += dt;
}

void scene::set_animation_time(time_ticks dt)
{
    auto reset_handler = [&](animated_node* n){
        n->restart();
        n->update(dt);
    };
    for(camera* c: cameras) reset_handler(c);
    for(animated_node* o: control_nodes) reset_handler(o);

    light_scene::visit_animated(reset_handler);
    mesh_scene::visit_animated(reset_handler);
    total_ticks = dt;
}

time_ticks scene::get_total_ticks() const
{
    return total_ticks;
}

bool scene::is_playing() const
{
    bool playing = false;
    auto check_playing = [&](animated_node* n){
        if(n->is_playing()) playing = true;
    };
    for(camera* c: cameras) check_playing(c);
    for(animated_node* o: control_nodes) check_playing(o);

    light_scene::visit_animated(check_playing);
    mesh_scene::visit_animated(check_playing);

    return playing;
}

vk::AccelerationStructureKHR scene::get_acceleration_structure(
    size_t device_index
) const {
    if(!ctx->is_ray_tracing_supported())
        throw std::runtime_error(
            "Trying to use TLAS, but ray tracing is not available!"
        );
    return *tlas->get_tlas_handle(device_index);
}

void scene::set_shadow_map_renderer(shadow_map_renderer* smr)
{
    this->smr = smr;
}

void scene::set_sh_grid_textures(
    std::unordered_map<sh_grid*, texture>* sh_grid_textures
){
    this->sh_grid_textures = sh_grid_textures;
}

vec2 scene::get_shadow_map_atlas_pixel_margin() const
{
    if(smr)
        return vec2(0.5)/vec2(smr->get_shadow_map_atlas()->get_size());
    else
        return vec2(0);
}

std::vector<descriptor_state> scene::get_descriptor_info(device_data* dev, int32_t camera_index) const
{
    auto& sb = scene_buffers[dev->index];
    const std::vector<sh_grid*>& sh_grids = get_sh_grids();

    std::vector<vk::DescriptorImageInfo> dii_3d;
    if(sh_grid_textures)
    {
        for(sh_grid* sg: sh_grids)
        {
            texture& tex = sh_grid_textures->at(sg);
            dii_3d.push_back({
                sb.sh_grid_sampler.get_sampler(dev->index),
                tex.get_image_view(dev->index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            });
        }
    }

    environment_map* envmap = get_environment_map();
    vk::AccelerationStructureKHR tlas = {};
    if(ctx->is_ray_tracing_supported())
        tlas = get_acceleration_structure(dev->index);

    std::vector<vk::DescriptorBufferInfo> dbi_vertex = get_vertex_buffer_bindings(dev->index);
    std::vector<vk::DescriptorBufferInfo> dbi_index = get_index_buffer_bindings(dev->index);

    std::vector<descriptor_state> descriptors = {
        {"scene", {*sb.scene_data, 0, VK_WHOLE_SIZE}},
        {"scene_metadata", {*sb.scene_metadata, 0, VK_WHOLE_SIZE}},
        {"vertices", dbi_vertex},
        {"indices", dbi_index},
        {"textures", sb.dii},
        {"directional_lights", {
            *sb.directional_light_data, 0, VK_WHOLE_SIZE
        }},
        {"point_lights", {*sb.point_light_data, 0, VK_WHOLE_SIZE}},
        {"tri_lights", {*sb.tri_light_data, 0, VK_WHOLE_SIZE}},
        {"environment_map_tex", {
            sb.envmap_sampler.get_sampler(dev->index),
            envmap ? envmap->get_image_view(dev->index) : vk::ImageView{},
            vk::ImageLayout::eShaderReadOnlyOptimal
        }},
        {"environment_map_alias_table", {
            envmap ? envmap->get_alias_table(dev->index) : vk::Buffer{}, 0, VK_WHOLE_SIZE
        }},
        {"textures3d", dii_3d},
        {"sh_grids", {*sb.sh_grid_data, 0, VK_WHOLE_SIZE}}
    };

    if(camera_index >= 0)
    {
        std::pair<size_t, size_t> camera_offset = sb.camera_data_offsets[camera_index];
        descriptors.push_back(
            {"camera", {*sb.camera_data, camera_offset.first, VK_WHOLE_SIZE}}
        );
    }

    if(ctx->is_ray_tracing_supported())
    {
        descriptors.push_back({"tlas", {1, this->tlas->get_tlas_handle(dev->index)}});
    }

    if(smr)
    {
        placeholders& pl = dev->ctx->get_placeholders();

        const atlas* shadow_map_atlas = smr->get_shadow_map_atlas();

        descriptors.push_back(
            {"shadow_maps", {*sb.shadow_map_data, 0, sb.shadow_map_range}}
        );
        descriptors.push_back(
            {"shadow_map_cascades", {
                *sb.shadow_map_data, sb.shadow_map_range,
                sb.shadow_map_cascade_range
            }}
        );
        descriptors.push_back(
            {"shadow_map_atlas", {
                pl.default_sampler.get_sampler(dev->index),
                shadow_map_atlas->get_image_view(dev->index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        );
        descriptors.push_back(
            {"shadow_map_atlas_test", {
                sb.shadow_sampler.get_sampler(dev->index),
                shadow_map_atlas->get_image_view(dev->index),
                vk::ImageLayout::eShaderReadOnlyOptimal
            }}
        );
    }
    return descriptors;
}

void scene::bind(basic_pipeline& pipeline, uint32_t frame_index, int32_t camera_index)
{
    device_data* dev = pipeline.get_device();
    std::vector<descriptor_state> descriptors = get_descriptor_info(dev, camera_index);
    pipeline.update_descriptor_set(descriptors, frame_index);
}

void scene::push(basic_pipeline& pipeline, vk::CommandBuffer cmd, int32_t camera_index)
{
    device_data* dev = pipeline.get_device();
    std::vector<descriptor_state> descriptors = get_descriptor_info(dev, camera_index);
    pipeline.push_descriptors(cmd, descriptors);
}

void scene::bind_placeholders(
    basic_pipeline& pipeline,
    size_t max_samplers,
    size_t max_3d_samplers
){
    device_data* dev = pipeline.get_device();
    placeholders& pl = dev->ctx->get_placeholders();

    pipeline.update_descriptor_set({
        {"textures", max_samplers},
        {"shadow_maps"},
        {"shadow_map_cascades"},
        {"shadow_map_atlas"},
        {"shadow_map_atlas_test", {
            pl.default_sampler.get_sampler(dev->index),
            pl.depth_test_sample.get_image_view(dev->index),
            vk::ImageLayout::eShaderReadOnlyOptimal
        }},
        {"textures3d", {
            pl.default_sampler.get_sampler(dev->index),
            pl.sample3d.get_image_view(dev->index),
            vk::ImageLayout::eShaderReadOnlyOptimal
        }, max_3d_samplers}
    });
}

void scene::init_acceleration_structures()
{
    if(!ctx->is_ray_tracing_supported()) return;

    uint32_t total_max_capacity = mesh_scene::get_max_capacity() + light_scene::get_max_capacity();
    tlas.emplace(*ctx, total_max_capacity);
}

scene::scene_buffer::scene_buffer(device_data& dev)
:   s_table(dev, true),
    scene_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    scene_metadata(dev, 0, vk::BufferUsageFlagBits::eUniformBuffer),
    directional_light_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    point_light_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    tri_light_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    sh_grid_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    shadow_map_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    camera_data(dev, 0, vk::BufferUsageFlagBits::eStorageBuffer),
    envmap_sampler(
        *dev.ctx, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0, true, false
    ),
    shadow_sampler(
        *dev.ctx,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0,
        true,
        false,
        true
    ),
    sh_grid_sampler(
        *dev.ctx,
        vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerMipmapMode::eNearest,
        0,
        true,
        false
    )
{
}

std::vector<uint32_t> get_viewport_reorder_mask(
    const std::set<int>& active_indices,
    size_t viewport_count
){
    std::vector<uint32_t> reorder;
    for(size_t i = 0; i < viewport_count; ++i)
    {
        if(active_indices.count(i))
            reorder.push_back(i);
    }
    for(size_t i = 0; i < viewport_count; ++i)
    {
        if(!active_indices.count(i))
            reorder.push_back(i);
    }
    return reorder;
}

}
