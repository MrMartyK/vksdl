#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("pipeline compiler test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_pipeline_compiler")
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

    std::filesystem::path shaderDir =
        std::filesystem::path(SDL_GetBasePath()) / "shaders";

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());
        assert(compiler.value().resolvedModel() == vksdl::PipelineModel::Monolithic);

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto handleResult = compiler.value().compile(builder);
        assert(handleResult.ok());
        assert(handleResult.value().isReady());
        assert(handleResult.value().isOptimized());

        std::printf("  ForceMonolithic compile: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        auto h = compiler.value().compile(builder);
        assert(h.ok());

        compiler.value().waitIdle();
        assert(compiler.value().pendingCount() == 0);

        std::printf("  pendingCount after waitIdle: ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto builder = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value());

        // First compile populates cache.
        auto h1 = compiler.value().compile(builder);
        assert(h1.ok());

        // Second compile should benefit from cache.
        auto h2 = compiler.value().compile(builder);
        assert(h2.ok());
        assert(h2.value().isOptimized());

        std::printf("  second compile (cache warm): ok\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        auto info = compiler.value().modelInfo();
        assert(info.model == vksdl::PipelineModel::Monolithic);
        // hasPCCC might be true or false depending on driver.
        std::printf("  modelInfo: model=Monolithic hasPCCC=%d hasGPL=%d fastLink=%d\n",
                    info.hasPCCC, info.hasGPL, info.fastLink);
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::Auto);
        assert(compiler.ok());

        auto resolved = compiler.value().resolvedModel();
        auto info = compiler.value().modelInfo();

        if (info.hasGPL && info.fastLink) {
            // May or may not use GPL depending on independent interpolation.
            std::printf("  Auto policy: resolved=%s (GPL available, fastLink=%d)\n",
                        resolved == vksdl::PipelineModel::GPL ? "GPL" : "Monolithic",
                        info.fastLink);
        } else {
            assert(resolved == vksdl::PipelineModel::Monolithic);
            std::printf("  Auto policy: resolved=Monolithic (no GPL or no fastLink)\n");
        }
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceShaderObject);
        assert(!compiler.ok());

        std::printf("  ForceShaderObject error: ok\n");
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

        // Wait for background optimization to complete.
        compiler.value().waitIdle();
        assert(compiler.value().pendingCount() == 0);

        // After waitIdle, the optimized pipeline should be available.
        assert(handle.isOptimized());
        std::printf("  GPL compile + optimize: ok (isOptimized=%d)\n",
                    handle.isOptimized());

        auto h2 = compiler.value().compile(builder);
        assert(h2.ok());
        assert(h2.value().isReady());

        compiler.value().waitIdle();
        std::printf("  GPL library caching: ok\n");

        auto builder2 = vksdl::PipelineBuilder(device.value())
            .vertexShader(shaderDir / "triangle.vert.spv")
            .fragmentShader(shaderDir / "triangle.frag.spv")
            .colorFormat(swapchain.value())
            .cullBack();

        auto h3 = compiler.value().compile(builder2);
        assert(h3.ok());
        assert(h3.value().vkPipeline() != handle.vkPipeline());

        compiler.value().waitIdle();
        std::printf("  GPL different builder: ok\n");
    } else {
        std::printf("  GPL tests: skipped (not available)\n");
    }

    {
        auto compiler = vksdl::PipelineCompiler::create(
            device.value(), cache, vksdl::PipelinePolicy::ForceMonolithic);
        assert(compiler.ok());

        std::vector<vksdl::PipelineHandle> handles;
        for (int i = 0; i < 5; ++i) {
            auto builder = vksdl::PipelineBuilder(device.value())
                .vertexShader(shaderDir / "triangle.vert.spv")
                .fragmentShader(shaderDir / "triangle.frag.spv")
                .colorFormat(swapchain.value());

            auto h = compiler.value().compile(builder);
            assert(h.ok());
            handles.push_back(std::move(h).value());
        }

        compiler.value().waitIdle();
        assert(compiler.value().pendingCount() == 0);

        for (auto& h : handles) {
            assert(h.isReady());
            assert(h.vkPipeline() != VK_NULL_HANDLE);
        }

        std::printf("  multiple concurrent compiles: ok\n");
    }

    device.value().waitIdle();
    std::printf("test_pipeline_compiler: all tests passed\n");
    return 0;
}
