#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("pipeline feedback test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_pipeline_feedback")
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

    std::filesystem::path shaderDir = std::filesystem::path(SDL_GetBasePath()) / "shaders";

    {
        auto result = vksdl::PipelineBuilder(device.value())
                          .vertexShader(shaderDir / "triangle.vert.spv")
                          .fragmentShader(shaderDir / "triangle.frag.spv")
                          .colorFormat(swapchain.value())
                          .build();
        assert(result.ok());

        auto* fb = result.value().feedback();
        assert(fb != nullptr && "feedback() should be non-null after build()");
        assert(fb->valid && "pipeline feedback should be valid");
        // Duration should be non-negative (some drivers report 0 for fast builds).
        assert(fb->durationMs >= 0.0);

        assert(fb->stages.size() == 2 && "should have vertex + fragment stages");
        assert(fb->stages[0].stage == VK_SHADER_STAGE_VERTEX_BIT);
        assert(fb->stages[1].stage == VK_SHADER_STAGE_FRAGMENT_BIT);

        std::printf("  graphics pipeline feedback: valid=%d cacheHit=%d "
                    "duration=%.3fms stages=%zu\n",
                    fb->valid, fb->cacheHit, fb->durationMs, fb->stages.size());
    }

    {
        auto cacheResult = vksdl::PipelineCache::create(device.value());
        assert(cacheResult.ok());
        auto& cache = cacheResult.value();

        // First build populates the cache.
        auto first = vksdl::PipelineBuilder(device.value())
                         .vertexShader(shaderDir / "triangle.vert.spv")
                         .fragmentShader(shaderDir / "triangle.frag.spv")
                         .colorFormat(swapchain.value())
                         .cache(cache)
                         .build();
        assert(first.ok());
        auto* fb1 = first.value().feedback();
        assert(fb1 != nullptr);

        // Second build should be a cache hit (driver-dependent, but expected).
        auto second = vksdl::PipelineBuilder(device.value())
                          .vertexShader(shaderDir / "triangle.vert.spv")
                          .fragmentShader(shaderDir / "triangle.frag.spv")
                          .colorFormat(swapchain.value())
                          .cache(cache)
                          .build();
        assert(second.ok());
        auto* fb2 = second.value().feedback();
        assert(fb2 != nullptr);

        // Some drivers don't reliably report cache hits, so just check validity.
        assert(fb2->valid);
        std::printf("  pipeline cache feedback: first=%.3fms second=%.3fms "
                    "secondCacheHit=%d\n",
                    fb1->durationMs, fb2->durationMs, fb2->cacheHit);
    }

    {
        // Some drivers report 0 for cached pipelines. Just verify we handle it.
        auto cacheResult = vksdl::PipelineCache::create(device.value());
        assert(cacheResult.ok());

        auto result = vksdl::PipelineBuilder(device.value())
                          .vertexShader(shaderDir / "triangle.vert.spv")
                          .fragmentShader(shaderDir / "triangle.frag.spv")
                          .colorFormat(swapchain.value())
                          .cache(cacheResult.value())
                          .build();
        assert(result.ok());
        auto* fb = result.value().feedback();
        assert(fb != nullptr);
        // 0.0 is valid, negative is not.
        assert(fb->durationMs >= 0.0);
        std::printf("  zero duration validity: ok (%.3fms)\n", fb->durationMs);
    }

    {
        auto result = vksdl::PipelineBuilder(device.value())
                          .vertexShader(shaderDir / "triangle.vert.spv")
                          .fragmentShader(shaderDir / "triangle.frag.spv")
                          .colorFormat(swapchain.value())
                          .build();
        assert(result.ok());

        auto pipeline = std::move(result).value();
        auto* fb1 = pipeline.feedback();
        assert(fb1 != nullptr && fb1->valid);
        double dur = fb1->durationMs;

        // Move-construct.
        vksdl::Pipeline moved(std::move(pipeline));
        auto* fb2 = moved.feedback();
        assert(fb2 != nullptr && fb2->valid);
        assert(fb2->durationMs == dur);

        std::printf("  move preserves stats: ok\n");
    }

    {
        auto result = vksdl::ComputePipelineBuilder(device.value())
                          .shader(shaderDir / "noop.comp.spv")
                          .build();
        assert(result.ok());

        auto* fb = result.value().feedback();
        assert(fb != nullptr);
        assert(fb->valid);
        assert(fb->stages.size() == 1);
        assert(fb->stages[0].stage == VK_SHADER_STAGE_COMPUTE_BIT);

        std::printf("  compute pipeline feedback: valid=%d stages=%zu "
                    "duration=%.3fms\n",
                    fb->valid, fb->stages.size(), fb->durationMs);
    }

    device.value().waitIdle();
    std::printf("test_pipeline_feedback: all tests passed\n");
    return 0;
}
