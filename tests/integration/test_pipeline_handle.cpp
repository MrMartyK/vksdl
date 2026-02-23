#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("pipeline handle test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_pipeline_handle")
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

    auto swapchain = vksdl::SwapchainBuilder(device.value(), surface.value())
        .size(window.value().pixelSize())
        .build();
    assert(swapchain.ok());

    auto cacheResult = vksdl::PipelineCache::create(device.value());
    assert(cacheResult.ok());
    auto& cache = cacheResult.value();

    auto frameSync = vksdl::FrameSync::create(device.value(), 2);
    assert(frameSync.ok());

    std::filesystem::path shaderDir =
        std::filesystem::path(SDL_GetBasePath()) / "shaders";

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto handleResult = compiler.value().compile(builder);
        assert(handleResult.ok());

        auto& handle = handleResult.value();
        assert(handle.isReady() && "handle should be ready after compile()");
        assert(handle.vkPipeline() != VK_NULL_HANDLE);
        assert(handle.vkPipelineLayout() != VK_NULL_HANDLE);

        // Monolithic path marks as optimized immediately.
        assert(handle.isOptimized());

        std::printf("  monolithic handle creation: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto handleResult = compiler.value().compile(builder);
        assert(handleResult.ok());

        // Record a command buffer with bind().
        auto frame = frameSync.value().nextFrame();
        assert(frame.ok());
        VkCommandBuffer cmd = frame.value().cmd;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        handleResult.value().bind(cmd);

        vkEndCommandBuffer(cmd);

        std::printf("  bind() command recording: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto handleResult = compiler.value().compile(builder);
        assert(handleResult.ok());

        VkPipeline origPipeline = handleResult.value().vkPipeline();

        // Move-construct.
        vksdl::PipelineHandle moved(std::move(handleResult.value()));
        assert(moved.isReady());
        assert(moved.vkPipeline() == origPipeline);

        // Moved-from handle is safe to destroy (no crash).
        std::printf("  move semantics: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceShaderObject);
        assert(!compiler.ok());
        assert(compiler.error().operation == "create pipeline compiler");

        std::printf("  ForceShaderObject error: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto builder1 = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto h1 = compiler.value().compile(builder1);
        assert(h1.ok());

        auto builder2 = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .cullBack();

        auto h2 = compiler.value().compile(builder2);
        assert(h2.ok());

        // Different pipelines.
        assert(h1.value().vkPipeline() != h2.value().vkPipeline());

        compiler.value().waitIdle();
        assert(compiler.value().pendingCount() == 0);

        std::printf("  multiple compiles: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());
        assert(compiler.value().resolvedModel() == vksdl::PipelineModel::Monolithic);
        assert(compiler.value().policy() == vksdl::PipelinePolicy::ForceMonolithic);

        std::printf("  resolvedModel/policy: ok\n");
    }

    if (device.value().hasGPL()) {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::PreferGPL);
        assert(compiler.ok());
        assert(compiler.value().resolvedModel() == vksdl::PipelineModel::GPL);

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto handleResult = compiler.value().compile(builder);
        assert(handleResult.ok());

        auto& handle = handleResult.value();
        assert(handle.isReady());
        assert(handle.vkPipeline() != VK_NULL_HANDLE);

        // Wait for background optimization.
        compiler.value().waitIdle();
        assert(handle.isOptimized() &&
               "handle should be optimized after waitIdle()");

        std::printf("  GPL compile + background optimize: ok\n");
    } else {
        std::printf("  GPL compile: skipped (not available)\n");
    }

    device.value().waitIdle();
    std::printf("test_pipeline_handle: all tests passed\n");
    return 0;
}
