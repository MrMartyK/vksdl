#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("pipeline cache test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_pipeline_cache")
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
        auto cache = vksdl::PipelineCache::create(device.value());
        assert(cache.ok() && "empty pipeline cache creation failed");
        assert(cache.value().vkPipelineCache() != VK_NULL_HANDLE);
        std::printf("  create empty cache: ok\n");
    }

    {
        auto cache = vksdl::PipelineCache::create(device.value());
        assert(cache.ok());

        auto pipeline = vksdl::PipelineBuilder(device.value())
                            .vertexShader(shaderDir / "triangle.vert.spv")
                            .fragmentShader(shaderDir / "triangle.frag.spv")
                            .colorFormat(swapchain.value())
                            .cache(cache.value())
                            .build();

        assert(pipeline.ok() && "pipeline build with cache failed");
        assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);

        // Cache should now contain data from the compiled pipeline.
        auto sz = cache.value().dataSize();
        assert(sz > 0 && "cache should contain data after pipeline build");
        std::printf("  pipeline build with cache: ok (cache size: %zu bytes)\n", sz);
    }

    {
        auto cache = vksdl::PipelineCache::create(device.value());
        assert(cache.ok());

        // Build a pipeline to populate the cache.
        auto p1 = vksdl::PipelineBuilder(device.value())
                      .vertexShader(shaderDir / "triangle.vert.spv")
                      .fragmentShader(shaderDir / "triangle.frag.spv")
                      .colorFormat(swapchain.value())
                      .cache(cache.value())
                      .build();
        assert(p1.ok());

        // Save to disk.
        std::filesystem::path cachePath =
            std::filesystem::path(SDL_GetBasePath()) / "test_pipeline.cache";
        auto saveResult = cache.value().save(cachePath);
        assert(saveResult.ok() && "cache save failed");
        assert(std::filesystem::exists(cachePath));
        auto fileSize = std::filesystem::file_size(cachePath);
        assert(fileSize > 0);
        std::printf("  save cache to disk: ok (%llu bytes)\n",
                    static_cast<unsigned long long>(fileSize));

        // Load from disk.
        auto loaded = vksdl::PipelineCache::load(device.value(), cachePath);
        assert(loaded.ok() && "cache load failed");
        assert(loaded.value().vkPipelineCache() != VK_NULL_HANDLE);
        assert(loaded.value().dataSize() > 0);
        std::printf("  load cache from disk: ok (size: %zu bytes)\n", loaded.value().dataSize());

        // Build again with loaded cache.
        auto p2 = vksdl::PipelineBuilder(device.value())
                      .vertexShader(shaderDir / "triangle.vert.spv")
                      .fragmentShader(shaderDir / "triangle.frag.spv")
                      .colorFormat(swapchain.value())
                      .cache(loaded.value())
                      .build();
        assert(p2.ok() && "pipeline build with loaded cache failed");
        std::printf("  pipeline build with loaded cache: ok\n");

        // Cleanup test file.
        std::filesystem::remove(cachePath);
    }

    {
        auto loaded = vksdl::PipelineCache::load(device.value(), "nonexistent_cache_file.bin");
        assert(loaded.ok() && "load from missing file should succeed with empty cache");
        assert(loaded.value().vkPipelineCache() != VK_NULL_HANDLE);
        std::printf("  load from missing file (fallback): ok\n");
    }

    {
        auto cache = vksdl::PipelineCache::create(device.value());
        assert(cache.ok());

        auto pipeline = vksdl::PipelineBuilder(device.value())
                            .vertexShader(shaderDir / "triangle.vert.spv")
                            .fragmentShader(shaderDir / "triangle.frag.spv")
                            .colorFormat(swapchain.value())
                            .cache(cache.value().vkPipelineCache())
                            .build();

        assert(pipeline.ok() && "pipeline with raw VkPipelineCache failed");
        std::printf("  cache escape hatch (VkPipelineCache): ok\n");
    }

    {
        auto dst = vksdl::PipelineCache::create(device.value());
        auto src = vksdl::PipelineCache::create(device.value());
        assert(dst.ok());
        assert(src.ok());

        // Build once into src so the merge has real data to merge.
        auto seeded = vksdl::PipelineBuilder(device.value())
                          .vertexShader(shaderDir / "triangle.vert.spv")
                          .fragmentShader(shaderDir / "triangle.frag.spv")
                          .colorFormat(swapchain.value())
                          .cache(src.value())
                          .build();
        assert(seeded.ok());
        assert(src.value().dataSize() > 0);

        auto merge1 = dst.value().merge(src.value());
        assert(merge1.ok());

        auto merge2 = dst.value().merge(src.value().vkPipelineCache());
        assert(merge2.ok());

        std::printf("  cache merge wrapper: ok\n");
    }

    {
        auto cache = vksdl::PipelineCache::create(device.value());
        assert(cache.ok());

        auto c1 = std::move(cache).value();
        assert(c1.vkPipelineCache() != VK_NULL_HANDLE);

        auto c2 = std::move(c1);
        assert(c2.vkPipelineCache() != VK_NULL_HANDLE);
        // c1 is moved-from -- destructor must not crash.
        std::printf("  move semantics: ok\n");
    }

    device.value().waitIdle();
    std::printf("pipeline cache test passed\n");
    return 0;
}
