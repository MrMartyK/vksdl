#include <vksdl/vksdl.hpp>
#include <vksdl/graph.hpp>
#include <vksdl/shader_reflect.hpp>

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>

using namespace vksdl::graph;

// Number of synthetic post-processing passes to stress-test graph scaling.
// 0 = original 6-pass deferred only. 34 = total 40 passes.
static constexpr std::uint32_t kSyntheticPasses = 34;

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Deferred Shading (Render Graph)", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_deferred")
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

    auto frames = vksdl::FrameSync::create(device, swapchain.imageCount()).value();
    auto allocator = vksdl::Allocator::create(instance, device).value();

    std::printf("GPU: %s\n", device.gpuName());

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    // Sampler for reading G-buffer / shadow / HDR textures.
    auto sampler = vksdl::SamplerBuilder(device)
        .linear()
        .clampToEdge()
        .build().value();

    // Reflect lighting and tonemap shaders for Layer 2 auto-bind.
    auto lightingRefl = vksdl::mergeReflections(
        vksdl::reflectSpvFile(shaderDir / "fullscreen.vert.spv",
                               VK_SHADER_STAGE_VERTEX_BIT).value(),
        vksdl::reflectSpvFile(shaderDir / "lighting.frag.spv",
                               VK_SHADER_STAGE_FRAGMENT_BIT).value()
    ).value();

    auto tonemapRefl = vksdl::mergeReflections(
        vksdl::reflectSpvFile(shaderDir / "fullscreen.vert.spv",
                               VK_SHADER_STAGE_VERTEX_BIT).value(),
        vksdl::reflectSpvFile(shaderDir / "tonemap.frag.spv",
                               VK_SHADER_STAGE_FRAGMENT_BIT).value()
    ).value();

    // Shadow: depth-only, no color attachments.
    auto shadowPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "fullscreen.vert.spv")
        .fragmentShader(shaderDir / "shadow.frag.spv")
        .depthFormat(VK_FORMAT_D32_SFLOAT)
        .build().value();

    // G-buffer: 2 color attachments (albedo RGBA8, normals RGBA16F) + depth.
    auto gbufferPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "fullscreen.vert.spv")
        .fragmentShader(shaderDir / "gbuffer.frag.spv")
        .colorFormat(VK_FORMAT_R8G8B8A8_UNORM)
        .colorFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
        .depthFormat(VK_FORMAT_D32_SFLOAT)
        .build().value();

    // Lighting: 1 color attachment (HDR RGBA16F), reads 4 textures via reflection.
    auto lightingPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "fullscreen.vert.spv")
        .fragmentShader(shaderDir / "lighting.frag.spv")
        .colorFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
        .reflectDescriptors()
        .build().value();

    // Tonemap: 1 color attachment (swapchain format), reads HDR via reflection.
    auto tonemapPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "fullscreen.vert.spv")
        .fragmentShader(shaderDir / "tonemap.frag.spv")
        .colorFormat(swapchain.format())
        .reflectDescriptors()
        .build().value();

    // UI: 1 color attachment (swapchain format), depth read-only.
    auto uiPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "fullscreen.vert.spv")
        .fragmentShader(shaderDir / "ui.frag.spv")
        .colorFormat(swapchain.format())
        .depthFormat(VK_FORMAT_D32_SFLOAT)
        .build().value();

    bool running = true;
    vksdl::Event event;
    std::uint32_t frameNumber = 0;
    double totalCompileUs = 0.0;

    constexpr std::uint32_t kShadowSize = 1024;

    // One graph per frame-in-flight. When acquireFrame returns frame index I,
    // the fence wait guarantees frame I's previous graph transients have
    // completed GPU execution, so we can safely reset and rebuild graph[I].
    std::vector<RenderGraph> graphs;
    graphs.reserve(swapchain.imageCount());
    for (std::uint32_t i = 0; i < swapchain.imageCount(); ++i)
        graphs.emplace_back(device, allocator);

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            device.waitIdle();
            for (auto& g : graphs) g.reset();
            (void)swapchain.recreate(device, window);

            // Recreate swapchain-dependent pipelines with new format.
            tonemapPipeline = vksdl::PipelineBuilder(device)
                .vertexShader(shaderDir / "fullscreen.vert.spv")
                .fragmentShader(shaderDir / "tonemap.frag.spv")
                .colorFormat(swapchain.format())
                .reflectDescriptors()
                .build().value();

            uiPipeline = vksdl::PipelineBuilder(device)
                .vertexShader(shaderDir / "fullscreen.vert.spv")
                .fragmentShader(shaderDir / "ui.frag.spv")
                .colorFormat(swapchain.format())
                .depthFormat(VK_FORMAT_D32_SFLOAT)
                .build().value();
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        // The fence wait inside acquireFrame guarantees this frame index's
        // previous GPU work has completed. Safe to reset graph transients now.
        auto& graph = graphs[frame.index];
        graph.reset();

        VkCommandBuffer cmd = frame.cmd;

        vksdl::beginOneTimeCommands(cmd);

        auto extent = swapchain.extent();

        ResourceState swapchainState{};
        swapchainState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto swapImg = graph.importImage(
            img.image, img.view, swapchain.format(),
            extent.width, extent.height,
            swapchainState, 1, 1, "swapchain");

        auto shadowDepth = graph.createImage({
            .width  = kShadowSize, .height = kShadowSize,
            .format = VK_FORMAT_D32_SFLOAT,
        }, "shadow_depth");

        auto gbufAlbedo = graph.createImage({
            .width = extent.width, .height = extent.height,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
        }, "gbuf_albedo");

        auto gbufNormals = graph.createImage({
            .width = extent.width, .height = extent.height,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        }, "gbuf_normals");

        auto gbufDepth = graph.createImage({
            .width = extent.width, .height = extent.height,
            .format = VK_FORMAT_D32_SFLOAT,
        }, "gbuf_depth");

        auto hdrColor = graph.createImage({
            .width = extent.width, .height = extent.height,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        }, "hdr_color");

        graph.addPass("shadow", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setDepthTarget(shadowDepth);
            },
            [&](PassContext& ctx, VkCommandBuffer c) {
                ctx.beginRendering(c);
                shadowPipeline.bind(c);
                vkCmdDraw(c, 3, 1, 0, 0);
                ctx.endRendering(c);
            });

        graph.addPass("gbuffer", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setColorTarget(0, gbufAlbedo);
                b.setColorTarget(1, gbufNormals);
                b.setDepthTarget(gbufDepth);
            },
            [&](PassContext& ctx, VkCommandBuffer c) {
                ctx.beginRendering(c);
                gbufferPipeline.bind(c);
                vkCmdDraw(c, 3, 1, 0, 0);
                ctx.endRendering(c);
            });

        graph.addPass("lighting", PassType::Graphics,
            lightingPipeline.vkPipeline(), lightingPipeline.vkPipelineLayout(), lightingRefl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, hdrColor);
                b.setSampler(sampler.vkSampler());
                b.bind("shadowDepth", shadowDepth);
                b.bind("gbufAlbedo",  gbufAlbedo);
                b.bind("gbufNormals", gbufNormals);
                b.bind("gbufDepth",   gbufDepth);
            },
            [&](PassContext& ctx, VkCommandBuffer c) {
                ctx.beginRendering(c);
                vkCmdDraw(c, 3, 1, 0, 0);
                ctx.endRendering(c);
            });

        graph.addPass("tonemap", PassType::Graphics,
            tonemapPipeline.vkPipeline(), tonemapPipeline.vkPipelineLayout(), tonemapRefl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, swapImg);
                b.setSampler(sampler.vkSampler());
                b.bind("hdrColor", hdrColor);
            },
            [&](PassContext& ctx, VkCommandBuffer c) {
                ctx.beginRendering(c);
                vkCmdDraw(c, 3, 1, 0, 0);
                ctx.endRendering(c);
            });

        graph.addPass("ui", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setColorTarget(0, swapImg, LoadOp::Load);
                b.setDepthTarget(gbufDepth, LoadOp::Load, DepthWrite::Disabled);
            },
            [&](PassContext& ctx, VkCommandBuffer c) {
                ctx.beginRendering(c);
                uiPipeline.bind(c);
                vkCmdDraw(c, 3, 1, 0, 0);
                ctx.endRendering(c);
            });

        // Synthetic post-processing chain: each pass reads one transient, writes another.
        std::vector<ResourceHandle> synthImages;
        ResourceHandle synthPrev = hdrColor;

        for (std::uint32_t si = 0; si < kSyntheticPasses; ++si) {
            char nameBuf[32];
            std::snprintf(nameBuf, sizeof(nameBuf), "synth_%u", si);
            auto synthImg = graph.createImage({
                .width = extent.width, .height = extent.height,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
            }, nameBuf);
            synthImages.push_back(synthImg);

            auto prev = synthPrev;
            auto cur  = synthImg;
            graph.addPass(nameBuf, PassType::Graphics,
                [prev, cur](PassBuilder& b) {
                    b.sampleImage(prev);
                    b.writeColorAttachment(cur);
                },
                [](PassContext&, VkCommandBuffer) {});

            synthPrev = synthImg;
        }

        ResourceState presentState{};
        presentState.lastWriteStage  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        presentState.currentLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        graph.addPass("present", PassType::Graphics,
            [&](PassBuilder& b) {
                b.access(swapImg, AccessType::Read, presentState);
            },
            [](PassContext&, VkCommandBuffer) {});

        auto compileResult = graph.compile();
        if (!compileResult.ok()) {
            std::fprintf(stderr, "Graph compile error: %s\n",
                         compileResult.error().message.c_str());
            break;
        }

        if (frameNumber == 0) graph.dumpLog();

        totalCompileUs += graph.stats().compileTimeUs;
        graph.execute(cmd);

        vksdl::endCommands(cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        frameNumber++;

        if (frameNumber % 100 == 0) {
            double avgUs = totalCompileUs / 100.0;
            const auto& s = graph.stats();
            std::printf("Frame %u: avg compile %.1fus (%u barriers, %u passes)\n",
                        frameNumber, avgUs,
                        s.imageBarrierCount + s.bufferBarrierCount,
                        s.passCount);
            totalCompileUs = 0.0;
        }
    }

    device.waitIdle();
    return 0;
}
