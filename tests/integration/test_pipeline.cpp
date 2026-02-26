#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("pipeline test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_pipeline")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
        .enableWindowSupport()
        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    auto device = vksdl::DeviceBuilder(instance.value(), surface.value())
        .graphicsDefaults()
        .preferDiscreteGpu()
        .build();
    assert(device.ok());

    auto swapchain = vksdl::SwapchainBuilder(device.value(), surface.value())
        .forWindow(window.value())
        .build();
    assert(swapchain.ok());

    std::filesystem::path shaderDir =
        std::filesystem::path(SDL_GetBasePath()) / "shaders";

    {
        auto code = vksdl::readSpv(shaderDir / "triangle.vert.spv");
        assert(code.ok() && "readSpv failed for valid shader");
        assert(!code.value().empty());
        std::printf("  readSpv valid file: ok\n");
    }

    {
        auto code = vksdl::readSpv("nonexistent.spv");
        assert(!code.ok());
        assert(code.error().operation == "read SPIR-V");
        std::printf("  readSpv invalid path: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .simpleColorPipeline(
                shaderDir / "triangle.vert.spv",
                shaderDir / "triangle.frag.spv",
                swapchain.value())
            .build();

        assert(result.ok() && "simpleColorPipeline preset failed");
        assert(result.value().vkPipeline() != VK_NULL_HANDLE);
        std::printf("  simpleColorPipeline preset: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .build();

        assert(result.ok() && "default pipeline creation failed");

        assert(result.value().vkPipeline() != VK_NULL_HANDLE);
        assert(result.value().vkPipelineLayout() != VK_NULL_HANDLE);
        std::printf("  default pipeline: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .cullBack()
            .clockwise()
            .enableBlending()
            .build();

        assert(result.ok() && "pipeline with convenience methods failed");
        std::printf("  pipeline (cullBack + cw + blending): ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .wireframe()
            .build();

        assert(result.ok() && "wireframe pipeline failed");
        std::printf("  wireframe pipeline: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
            .build();

        assert(result.ok() && "triangle strip pipeline failed");
        std::printf("  triangle strip topology: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .build();

        assert(!result.ok());
        auto msg = result.error().format();
        assert(msg.find("vertex shader") != std::string::npos);
        std::printf("  missing vertex shader rejected: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .build();

        assert(!result.ok());
        auto msg = result.error().format();
        assert(msg.find("color format") != std::string::npos);
        std::printf("  missing color format rejected: ok\n");
    }

    {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        range.offset = 0;
        range.size = 64;

        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .pushConstantRange(range)
            .build();

        assert(result.ok() && "pipeline with push constants failed");
        std::printf("  pipeline with push constants: ok\n");
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .build();
        assert(result.ok());

        auto p1 = std::move(result).value();
        assert(p1.vkPipeline() != VK_NULL_HANDLE);

        auto p2 = std::move(p1);
        assert(p2.vkPipeline() != VK_NULL_HANDLE);
        // p1 is moved-from -- its destructor must not crash
        std::printf("  move semantics: ok\n");
    }

    {
        auto r1 = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .build();

        auto r2 = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value().format())
            .build();

        assert(r1.ok() && r2.ok());
        std::printf("  colorFormat(swapchain) overload: ok\n");
    }

    device.value().waitIdle();
    std::printf("pipeline test passed\n");
    return 0;
}
