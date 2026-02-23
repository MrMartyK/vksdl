#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

#include <SDL3/SDL.h>

static std::filesystem::path shaderDir() {
    return std::filesystem::path(SDL_GetBasePath()) / "shaders";
}

int main() {
    auto app      = vksdl::App::create().value();
    auto window   = app.createWindow("test", 64, 64).value();
    auto instance = vksdl::InstanceBuilder{}
        .appName("test_compute")
        .requireVulkan(1, 3)
        .enableWindowSupport()
        .build().value();
    auto surface  = vksdl::Surface::create(instance, window).value();
    auto device   = vksdl::DeviceBuilder(instance, surface)
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .build().value();

    // 1. Create compute pipeline from shader path
    {
        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir() / "noop.comp.spv")
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  compute pipeline from path: ok\n");
    }

    // 2. Pipeline layout created internally
    {
        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir() / "noop.comp.spv")
            .build().value();
        assert(pipeline.vkPipelineLayout() != VK_NULL_HANDLE);
        std::printf("  internal layout: ok\n");
    }

    // 3. External pipeline layout
    {
        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkPipelineLayout externalLayout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device.vkDevice(), &layoutCI, nullptr, &externalLayout);
        assert(externalLayout != VK_NULL_HANDLE);

        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir() / "noop.comp.spv")
            .pipelineLayout(externalLayout)
            .build().value();
        assert(pipeline.vkPipelineLayout() == externalLayout);
        std::printf("  external layout: ok\n");

        // Pipeline doesn't own the layout, so we must destroy it.
        vkDestroyPipelineLayout(device.vkDevice(), externalLayout, nullptr);
    }

    // 4. Push constant range
    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset     = 0;
        range.size       = sizeof(float);

        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir() / "noop.comp.spv")
            .pushConstantRange(range)
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  push constant range: ok\n");
    }

    // 5. Missing shader rejected
    {
        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader("nonexistent.comp.spv")
            .build();
        assert(!pipeline.ok());
        assert(pipeline.error().message.find("nonexistent") != std::string::npos);
        std::printf("  missing shader rejected: ok\n");
    }

    // 6. No shader set rejected
    {
        auto pipeline = vksdl::ComputePipelineBuilder(device).build();
        assert(!pipeline.ok());
        std::printf("  no shader rejected: ok\n");
    }

    // 7. Move semantics
    {
        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir() / "noop.comp.spv")
            .build().value();

        vksdl::Pipeline moved = std::move(pipeline);
        assert(moved.vkPipeline() != VK_NULL_HANDLE);
        assert(pipeline.vkPipeline() == VK_NULL_HANDLE);  // NOLINT moved-from
        std::printf("  move semantics: ok\n");
    }

    // 8. Descriptor set layout
    {
        VkDescriptorSetLayoutCreateInfo dslCI{};
        dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(device.vkDevice(), &dslCI, nullptr, &dsl);

        auto pipeline = vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir() / "noop.comp.spv")
            .descriptorSetLayout(dsl)
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  descriptor set layout: ok\n");

        vkDestroyDescriptorSetLayout(device.vkDevice(), dsl, nullptr);
    }

    device.waitIdle();
    std::printf("all compute pipeline tests passed\n");
    return 0;
}
