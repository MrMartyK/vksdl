#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("descriptor_set test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_descriptor_set")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
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

    auto allocator = vksdl::Allocator::create(instance.value(), device.value());
    assert(allocator.ok());

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
            .build();
        assert(ds.ok());
        assert(ds.value().vkDescriptorSet() != VK_NULL_HANDLE);
        assert(ds.value().vkDescriptorSetLayout() != VK_NULL_HANDLE);
        assert(ds.value().vkDescriptorPool() != VK_NULL_HANDLE);
        std::printf("  storage image binding: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .build();
        assert(ds.ok());
        assert(ds.value().vkDescriptorSet() != VK_NULL_HANDLE);
        assert(ds.value().vkDescriptorSetLayout() != VK_NULL_HANDLE);
        assert(ds.value().vkDescriptorPool() != VK_NULL_HANDLE);
        std::printf("  uniform buffer binding: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .addCombinedImageSampler(1, VK_SHADER_STAGE_FRAGMENT_BIT)
            .build();
        assert(ds.ok());
        assert(ds.value().vkDescriptorSet() != VK_NULL_HANDLE);
        std::printf("  multiple bindings: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value()).build();
        assert(!ds.ok());
        std::printf("  empty builder rejected: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addStorageBuffer(0, VK_SHADER_STAGE_COMPUTE_BIT)
            .build();
        assert(ds.ok());
        auto d1 = std::move(ds).value();
        auto d2 = std::move(d1);
        assert(d2.vkDescriptorSet() != VK_NULL_HANDLE);
        assert(d1.vkDescriptorSet() == VK_NULL_HANDLE);
        std::printf("  move semantics: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .build();
        assert(ds.ok());

        auto buf = vksdl::BufferBuilder(allocator.value())
            .size(64)
            .uniformBuffer()
            .build();
        assert(buf.ok());

        ds.value().updateBuffer(0, buf.value().vkBuffer(), 64);
        std::printf("  updateBuffer: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
            .build();
        assert(ds.ok());

        auto img = vksdl::ImageBuilder(allocator.value())
            .size(64, 64)
            .format(VK_FORMAT_R8G8B8A8_UNORM)
            .storage()
            .build();
        assert(img.ok());

        ds.value().updateImage(0, img.value().vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        std::printf("  updateImage: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        VK_SHADER_STAGE_COMPUTE_BIT, 1)
            .build();
        assert(ds.ok());
        assert(ds.value().vkDescriptorSet() != VK_NULL_HANDLE);
        std::printf("  escape hatch addBinding: ok\n");
    }

    {
        auto ds = vksdl::DescriptorSetBuilder(device.value())
            .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
            .build();
        assert(ds.ok());

        std::filesystem::path shaderDir =
            std::filesystem::path(SDL_GetBasePath()) / "shaders";

        auto pipeline = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(VK_FORMAT_B8G8R8A8_SRGB)
            .descriptorSetLayout(ds.value().vkDescriptorSetLayout())
            .build();
        assert(pipeline.ok());
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  layout with PipelineBuilder: ok\n");
    }

    device.value().waitIdle();
    std::printf("all descriptor_set tests passed\n");
    return 0;
}
