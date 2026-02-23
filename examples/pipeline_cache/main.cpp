#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_scancode.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main() {
    auto app    = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Pipeline Cache", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_pipeline_cache")
        .requireVulkan(1, 3)
        .enableWindowSupport()
        .build().value();

    auto surface = vksdl::Surface::create(instance, window).value();

    auto device = vksdl::DeviceBuilder(instance, surface)
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .preferDiscreteGpu()
        .build().value();

    auto swapchain = vksdl::SwapchainBuilder(device, surface)
        .size(window.pixelSize())
        .build().value();

    auto frames    = vksdl::FrameSync::create(device, swapchain.imageCount()).value();
    auto allocator = vksdl::Allocator::create(instance, device).value();

    auto storageImage = vksdl::ImageBuilder(allocator)
        .size(swapchain.extent().width, swapchain.extent().height)
        .format(VK_FORMAT_R8G8B8A8_UNORM)
        .storage()
        .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .build().value();

    auto descriptors = vksdl::DescriptorSetBuilder(device)
        .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    descriptors.updateImage(0, storageImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";
    std::filesystem::path cachePath = vksdl::exeDir() / "pipeline.cache";

    auto cache       = vksdl::PipelineCache::load(device, cachePath).value();
    bool cacheLoaded = std::filesystem::exists(cachePath);
    std::printf("Pipeline cache %s (initial size: %zu bytes)\n",
                cacheLoaded ? "loaded from disk" : "created fresh",
                cache.dataSize());

    using Clock = std::chrono::high_resolution_clock;

    static constexpr int kPresetCount = 4;

    const char* kNames[kPresetCount] = {
        "Mandelbrot", "Plasma", "SDF Sphere", "Voronoi"
    };
    const char* kFiles[kPresetCount] = {
        "mandelbrot.comp.spv",
        "plasma.comp.spv",
        "sdf_sphere.comp.spv",
        "voronoi.comp.spv"
    };

    auto buildOne = [&](int i) -> vksdl::Pipeline {
        return vksdl::ComputePipelineBuilder(device)
            .shader(shaderDir / kFiles[i])
            .descriptorSetLayout(descriptors.vkDescriptorSetLayout())
            .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
            .cache(cache)
            .build().value();
    };

    double coldMs[kPresetCount] = {};
    std::vector<vksdl::Pipeline> pipelines;
    pipelines.reserve(kPresetCount);

    std::printf("Cold builds:\n");
    for (int i = 0; i < kPresetCount; ++i) {
        auto t0   = Clock::now();
        pipelines.push_back(buildOne(i));
        auto t1   = Clock::now();
        coldMs[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  %-12s  %.2f ms\n", kNames[i], coldMs[i]);
    }

    // Persist the populated cache so warm builds can reuse it.
    (void)cache.save(cachePath);

    // Rebuild all four pipelines with the populated cache to measure warm time.
    double warmMs[kPresetCount] = {};

    std::printf("Warm builds (cached):\n");
    for (int i = 0; i < kPresetCount; ++i) {
        auto t0 = Clock::now();
        // Build and immediately discard -- we render with the cold pipelines.
        vksdl::Pipeline warm = buildOne(i);
        auto t1   = Clock::now();
        warmMs[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  %-12s  %.2f ms\n", kNames[i], warmMs[i]);
        // warm destroyed here
    }

    std::printf("Cache size: %zu bytes\n", cache.dataSize());
    std::printf("Press 1-4 to switch presets, ESC to quit\n");

    // Title update helper: shows preset name, cold/warm times, and speedup ratio.
    auto updateTitle = [&](int preset) {
        double speedup = coldMs[preset] / std::max(warmMs[preset], 0.001);
        char   buf[256];
        std::snprintf(buf, sizeof(buf),
            "Pipeline Cache | %s | Cold: %.1f ms  Cached: %.1f ms  (%.0fx faster) | Cache: %.1f KB",
            kNames[preset],
            coldMs[preset],
            warmMs[preset],
            speedup,
            static_cast<double>(cache.dataSize()) / 1024.0);
        SDL_SetWindowTitle(window.sdlWindow(), buf);
    };

    int  currentPreset = 0;
    bool running       = true;
    auto startTime     = Clock::now();
    vksdl::Event event;

    updateTitle(currentPreset);

    int frameNum = 0;
    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
            if (event.type == vksdl::EventType::KeyDown) {
                if (event.key == SDL_SCANCODE_ESCAPE) {
                    running = false;
                } else if (event.key == SDL_SCANCODE_1) {
                    currentPreset = 0; updateTitle(currentPreset);
                } else if (event.key == SDL_SCANCODE_2) {
                    currentPreset = 1; updateTitle(currentPreset);
                } else if (event.key == SDL_SCANCODE_3) {
                    currentPreset = 2; updateTitle(currentPreset);
                } else if (event.key == SDL_SCANCODE_4) {
                    currentPreset = 3; updateTitle(currentPreset);
                }
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);

            storageImage = vksdl::ImageBuilder(allocator)
                .size(swapchain.extent().width, swapchain.extent().height)
                .format(VK_FORMAT_R8G8B8A8_UNORM)
                .storage()
                .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                .build().value();

            descriptors.updateImage(0, storageImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        float elapsed = std::chrono::duration<float>(Clock::now() - startTime).count();

        VkCommandBuffer cmd = frame.cmd;
        vksdl::beginOneTimeCommands(cmd);

        vksdl::transitionToComputeWrite(cmd, storageImage.vkImage());

        auto& pipe = pipelines[static_cast<std::size_t>(currentPreset)];
        pipe.bind(cmd, descriptors);
        pipe.pushConstants(cmd, elapsed);

        std::uint32_t groupsX = (swapchain.extent().width  + 15) / 16;
        std::uint32_t groupsY = (swapchain.extent().height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vksdl::blitToSwapchain(cmd,
                               storageImage,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               img.image, swapchain.extent());

        vksdl::endCommands(cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        ++frameNum;
        if (frameNum % 60 == 0) {
            double speedup = coldMs[currentPreset] / std::max(warmMs[currentPreset], 0.001);
            std::printf("[frame %d] preset=%s cold=%.2fms warm=%.2fms speedup=%.0fx cache=%zuB\n",
                        frameNum, kNames[currentPreset],
                        coldMs[currentPreset], warmMs[currentPreset],
                        speedup, cache.dataSize());
        }
    }

    device.waitIdle();

    // Persist the final cache state on clean exit.
    (void)cache.save(cachePath);

    return 0;
}
