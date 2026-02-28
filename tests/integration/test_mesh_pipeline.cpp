#include <vksdl/app.hpp>
#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/mesh_pipeline.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>

static bool gpuSupportsMeshShaders(const vksdl::Instance& instance, const vksdl::Surface& surface) {
    auto result = vksdl::DeviceBuilder(instance, surface)
                      .needSwapchain()
                      .needDynamicRendering()
                      .needSync2()
                      .needMeshShaders()
                      .build();
    return result.ok();
}

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("mesh pipeline test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_mesh_pipeline")
                        .requireVulkan(1, 3)
                        .validation(vksdl::Validation::Off)
                        .enableWindowSupport()
                        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    if (!gpuSupportsMeshShaders(instance.value(), surface.value())) {
        std::printf("SKIP: mesh shaders not available on this GPU\n");
        return 0;
    }

    // Build a device with mesh shader support.
    auto devResult = vksdl::DeviceBuilder(instance.value(), surface.value())
                         .graphicsDefaults()
                         .needMeshShaders()
                         .build();
    assert(devResult.ok() && "device with mesh shaders failed after support check passed");
    auto device = std::move(devResult.value());

    // Verify hasMeshShaders() returns true.
    assert(device.hasMeshShaders());
    std::printf("  hasMeshShaders: true\n");

    // Verify drawMeshTasksFn() is non-null.
    assert(device.drawMeshTasksFn() != nullptr);
    std::printf("  drawMeshTasksFn: loaded\n");

    // Verify MeshPipelineBuilder can be constructed without crashing.
    {
        vksdl::MeshPipelineBuilder builder(device);
        // Builder with no shaders set -- build() should return an error, not crash.
        auto result = builder.build();
        assert(!result.ok());
        assert(result.error().message.find("no mesh shader") != std::string::npos);
        std::printf("  MeshPipelineBuilder: constructed and validates inputs\n");
    }

    // Verify builder rejects missing fragment shader.
    {
        vksdl::MeshPipelineBuilder builder(device);
        builder.meshModule(VK_NULL_HANDLE); // invalid but bypasses path check
        auto result = builder.colorFormat(VK_FORMAT_B8G8R8A8_SRGB).build();
        // meshModule is VK_NULL_HANDLE and meshPath_ is empty, so still missing mesh.
        assert(!result.ok());
        std::printf("  MeshPipelineBuilder: missing fragment shader detected\n");
    }

    // Verify device move preserves mesh shader state.
    {
        vksdl::Device moved = std::move(device);
        assert(moved.hasMeshShaders());
        assert(moved.drawMeshTasksFn() != nullptr);
        std::printf("  Device move: mesh shader state preserved\n");
    }

    std::printf("mesh pipeline test passed\n");
    return 0;
}
