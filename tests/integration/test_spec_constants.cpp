#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("spec constants test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_spec_constants")
        .requireVulkan(1, 3)
        .enableWindowSupport()
        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    auto device = vksdl::DeviceBuilder(instance.value(), surface.value())
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .preferDiscreteGpu()
        .build();
    assert(device.ok());

    // Compute pipeline with specConstant
    {
        auto pipeline = vksdl::ComputePipelineBuilder(device.value())
            .shader("shaders/spec_const.comp.spv")
            .specConstant(0, std::uint32_t{16})   // override local_size_x
            .specConstant(1, 2.5f)                 // override SCALE
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  compute spec constants: ok\n");
    }

    // Compute pipeline with raw VkSpecializationInfo escape hatch
    {
        std::uint32_t localSizeX = 32;
        VkSpecializationMapEntry entry{};
        entry.constantID = 0;
        entry.offset     = 0;
        entry.size       = sizeof(localSizeX);

        VkSpecializationInfo specInfo{};
        specInfo.mapEntryCount = 1;
        specInfo.pMapEntries   = &entry;
        specInfo.dataSize      = sizeof(localSizeX);
        specInfo.pData         = &localSizeX;

        auto pipeline = vksdl::ComputePipelineBuilder(device.value())
            .shader("shaders/spec_const.comp.spv")
            .specialize(specInfo)
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  compute specialize escape hatch: ok\n");
    }

    // Graphics pipeline with spec constants (noop -- just tests the plumbing)
    {
        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader("shaders/triangle.vert.spv")
            .fragmentShader("shaders/triangle.frag.spv")
            .colorFormat(VK_FORMAT_B8G8R8A8_SRGB)
            .specConstant(0, 1u)
            .build();
        // Spec constants for IDs that don't exist in the shader are ignored
        // by the driver (Vulkan spec: unmatched entries are silently skipped).
        assert(pipeline.ok());
        std::printf("  graphics spec constants: ok\n");
    }

    device.value().waitIdle();
    std::printf("spec constants test passed\n");
    return 0;
}
