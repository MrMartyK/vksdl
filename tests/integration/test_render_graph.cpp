#include <vksdl/vksdl.hpp>
#include <vksdl/graph.hpp>
#include <vksdl/shader_reflect.hpp>

#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace vksdl::graph;

// Helper: create a one-shot command pool + command buffer, record, submit, wait.
struct OneShotCmd {
    VkDevice      device = VK_NULL_HANDLE;
    VkCommandPool pool   = VK_NULL_HANDLE;
    VkCommandBuffer cmd  = VK_NULL_HANDLE;

    static OneShotCmd begin(VkDevice device, std::uint32_t queueFamily) {
        OneShotCmd c;
        c.device = device;

        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.queueFamilyIndex = queueFamily;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        auto vr = vkCreateCommandPool(device, &poolCI, nullptr, &c.pool);
        assert(vr == VK_SUCCESS);

        VkCommandBufferAllocateInfo allocCI{};
        allocCI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocCI.commandPool        = c.pool;
        allocCI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocCI.commandBufferCount = 1;
        vr = vkAllocateCommandBuffers(device, &allocCI, &c.cmd);
        assert(vr == VK_SUCCESS);

        VkCommandBufferBeginInfo beginCI{};
        beginCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vr = vkBeginCommandBuffer(c.cmd, &beginCI);
        assert(vr == VK_SUCCESS);

        return c;
    }

    void submitAndWait(VkQueue queue) {
        auto vr = vkEndCommandBuffer(cmd);
        assert(vr == VK_SUCCESS);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        vr = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        assert(vr == VK_SUCCESS);
        vr = vkQueueWaitIdle(queue);
        assert(vr == VK_SUCCESS);

        vkDestroyCommandPool(device, pool, nullptr);
    }
};

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("render graph test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_render_graph")
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

    VkDevice vkDev = device.value().vkDevice();
    VkQueue  queue = device.value().graphicsQueue();
    std::uint32_t queueFamily = device.value().queueFamilies().graphics;

    // Create a test image for import tests.
    auto testImage = vksdl::ImageBuilder(allocator.value())
        .size(64, 64)
        .format(VK_FORMAT_R8G8B8A8_UNORM)
        .colorAttachment()
        .build();
    assert(testImage.ok());

    std::printf("render graph test\n");

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto img = graph.importImage(testImage.value(), initState);
        assert(img.valid());

        bool recorded = false;
        graph.addPass("draw", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.vkImage(img) == testImage.value().vkImage());
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());
        assert(graph.isCompiled());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        std::printf("  minimal graph: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        int order = 0;
        int aOrder = -1, bOrder = -1;

        graph.addPass("writePass", PassType::Compute,
            [&](PassBuilder& b) { b.writeStorageImage(img); },
            [&](PassContext&, VkCommandBuffer) { aOrder = order++; });

        graph.addPass("readPass", PassType::Compute,
            [&](PassBuilder& b) { b.readStorageImage(img); },
            [&](PassContext&, VkCommandBuffer) { bOrder = order++; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aOrder == 0 && "write pass should execute first");
        assert(bOrder == 1 && "read pass should execute second");
        std::printf("  two-pass dependency: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{};
        desc.width     = 32;
        desc.height    = 32;
        desc.format    = VK_FORMAT_R8G8B8A8_UNORM;
        desc.mipLevels = 1;

        auto transient = graph.createImage(desc);

        bool aRecorded = false, bRecorded = false;

        graph.addPass("write", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(transient); },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.vkImage(transient) != VK_NULL_HANDLE);
                assert(ctx.vkImageView(transient) != VK_NULL_HANDLE);
                aRecorded = true;
            });

        graph.addPass("read", PassType::Graphics,
            [&](PassBuilder& b) { b.sampleImage(transient); },
            [&](PassContext&, VkCommandBuffer) { bRecorded = true; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aRecorded && bRecorded);
        std::printf("  transient image: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        BufferDesc desc{};
        desc.size = 1024;
        auto buf = graph.createBuffer(desc);

        bool aRecorded = false, bRecorded = false;

        graph.addPass("compute", PassType::Compute,
            [&](PassBuilder& b) { b.writeStorageBuffer(buf); },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.vkBuffer(buf) != VK_NULL_HANDLE);
                assert(ctx.bufferSize(buf) == 1024);
                aRecorded = true;
            });

        graph.addPass("draw", PassType::Graphics,
            [&](PassBuilder& b) { b.readVertexBuffer(buf); },
            [&](PassContext&, VkCommandBuffer) { bRecorded = true; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aRecorded && bRecorded);
        std::printf("  transient buffer: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc descA{};
        descA.width = 16; descA.height = 16;
        descA.format = VK_FORMAT_R8G8B8A8_UNORM;
        auto imgA = graph.createImage(descA);
        auto imgB = graph.createImage(descA);

        bool aRecorded = false, bRecorded = false;

        graph.addPass("passA", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(imgA); },
            [&](PassContext&, VkCommandBuffer) { aRecorded = true; });

        graph.addPass("passB", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(imgB); },
            [&](PassContext&, VkCommandBuffer) { bRecorded = true; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aRecorded && bRecorded);
        std::printf("  independent passes: ok\n");
    }

    // Ping-pong pattern: A writes X + reads Y, B writes Y + reads X.
    // RAW and WAR edges both point forward in declaration order, so this
    // resolves to A-then-B (not a cycle).
    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{};
        desc.width = 16; desc.height = 16;
        desc.format = VK_FORMAT_R8G8B8A8_UNORM;
        auto imgX = graph.createImage(desc);
        auto imgY = graph.createImage(desc);

        bool aRecorded = false, bRecorded = false;
        graph.addPass("A", PassType::Compute,
            [&](PassBuilder& b) {
                b.writeStorageImage(imgX);
                b.readStorageImage(imgY);
            },
            [&](PassContext&, VkCommandBuffer) { aRecorded = true; });

        graph.addPass("B", PassType::Compute,
            [&](PassBuilder& b) {
                b.writeStorageImage(imgY);
                b.readStorageImage(imgX);
            },
            [&](PassContext&, VkCommandBuffer) { bRecorded = true; });

        auto r = graph.compile();
        assert(r.ok() && "ping-pong should compile (declaration order resolves it)");

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aRecorded && bRecorded);
        std::printf("  ping-pong ordering: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        int count = 0;
        graph.addPass("draw", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) { count++; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);
        assert(count == 1);

        graph.reset();
        assert(!graph.isCompiled());

        img = graph.importImage(testImage.value(), initState);
        graph.addPass("draw2", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) { count++; });

        r = graph.compile();
        assert(r.ok());

        oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);
        assert(count == 2);

        std::printf("  reset and rebuild: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        ImageDesc transientDesc{};
        transientDesc.width = 32; transientDesc.height = 32;
        transientDesc.format = VK_FORMAT_R8G8B8A8_UNORM;
        auto transient = graph.createImage(transientDesc);

        graph.addPass("write", PassType::Graphics,
            [&](PassBuilder& b) {
                b.writeColorAttachment(img);
                b.writeColorAttachment(transient);
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                // Imported image resolves to the real VkImage.
                assert(ctx.vkImage(img) == testImage.value().vkImage());
                assert(ctx.vkImageView(img) == testImage.value().vkImageView());

                // Transient image has valid handles.
                assert(ctx.vkImage(transient) != VK_NULL_HANDLE);
                assert(ctx.vkImageView(transient) != VK_NULL_HANDLE);

                // Metadata works.
                assert(ctx.imageFormat(transient) == VK_FORMAT_R8G8B8A8_UNORM);
                auto ext = ctx.imageExtent(transient);
                assert(ext.width == 32 && ext.height == 32);
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        std::printf("  PassContext resolution: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        // Pass A writes, then overrides the tracked state.
        graph.addPass("passA", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext& ctx, VkCommandBuffer) {
                ResourceState overrideState{};
                overrideState.lastWriteStage  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                overrideState.lastWriteAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                overrideState.currentLayout   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                ctx.assumeState(img, overrideState);
            });

        // Pass B reads -- the barrier should use the overridden state as source.
        bool bRecorded = false;
        graph.addPass("passB", PassType::Graphics,
            [&](PassBuilder& b) { b.sampleImage(img); },
            [&](PassContext&, VkCommandBuffer) { bRecorded = true; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(bRecorded);
        std::printf("  assumeState escape hatch: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        // Pass A: write as color attachment.
        graph.addPass("render", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [](PassContext&, VkCommandBuffer) {});

        // Pass B: transition to PRESENT_SRC via raw access escape hatch.
        ResourceState presentState{};
        presentState.lastWriteStage       = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        presentState.readAccessSinceWrite = VK_ACCESS_2_NONE;
        presentState.currentLayout        = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        graph.addPass("present", PassType::Graphics,
            [&](PassBuilder& b) {
                b.access(img, AccessType::Read, presentState);
            },
            [](PassContext&, VkCommandBuffer) {});

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        std::printf("  swapchain workflow: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        int order = 0;
        int aO = -1, bO = -1, cO = -1;

        graph.addPass("write", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) { aO = order++; });

        graph.addPass("readCompute", PassType::Compute,
            [&](PassBuilder& b) { b.readStorageImage(img); },
            [&](PassContext&, VkCommandBuffer) { bO = order++; });

        graph.addPass("readFrag", PassType::Graphics,
            [&](PassBuilder& b) { b.sampleImage(img); },
            [&](PassContext&, VkCommandBuffer) { cO = order++; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aO == 0 && "writer must execute first");
        assert(bO >= 1 && cO >= 1);
        std::printf("  multi-reader correctness: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        VkImageLayout capturedLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool writerRan = false;
        bool readerRan = false;

        graph.addPass("writer26", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) { writerRan = true; });

        graph.addPass("reader26", PassType::Graphics,
            [&](PassBuilder& b) { b.sampleImage(img); },
            [&](PassContext& ctx, VkCommandBuffer) {
                capturedLayout = ctx.imageLayout(img);
                readerRan = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(writerRan);
        assert(readerRan);
        assert(capturedLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        std::printf("  PassContext::imageLayout(): ok\n");
    }


    // Create a depth image for depth target tests.
    auto depthImage = vksdl::ImageBuilder(allocator.value())
        .size(64, 64)
        .format(VK_FORMAT_D32_SFLOAT)
        .depthAttachment()
        .build();
    assert(depthImage.ok());

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        bool recorded = false;
        graph.addPass("layer1draw", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setColorTarget(0, img);
            },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.hasRenderTargets());
                ctx.beginRendering(cmd);
                assert(ctx.renderingActive());
                ctx.endRendering(cmd);
                assert(!ctx.renderingActive());
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        std::printf("  Layer 1 basic setColorTarget: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto depth = graph.importImage(depthImage.value(), initState);

        bool recorded = false;
        graph.addPass("depthWrite", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setDepthTarget(depth);
            },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.hasRenderTargets());
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        std::printf("  Layer 1 setDepthTarget (Enabled): ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto depth = graph.importImage(depthImage.value(), initState);

        int order = 0;
        int aOrder = -1, bOrder = -1;

        graph.addPass("depthProducer", PassType::Graphics,
            [&](PassBuilder& b) { b.writeDepthAttachment(depth); },
            [&](PassContext&, VkCommandBuffer) { aOrder = order++; });

        graph.addPass("depthConsumer", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setDepthTarget(depth, LoadOp::Load, DepthWrite::Disabled);
            },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.hasRenderTargets());
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                bOrder = order++;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aOrder == 0 && "depth producer must execute first");
        assert(bOrder == 1 && "depth consumer must execute second");
        std::printf("  Layer 1 setDepthTarget (Disabled): ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc colorDesc{};
        colorDesc.width = 32; colorDesc.height = 32;
        colorDesc.format = VK_FORMAT_R8G8B8A8_UNORM;

        ImageDesc depthDesc{};
        depthDesc.width = 32; depthDesc.height = 32;
        depthDesc.format = VK_FORMAT_D32_SFLOAT;

        auto color0 = graph.createImage(colorDesc);
        auto color1 = graph.createImage(colorDesc);
        auto depth  = graph.createImage(depthDesc);

        bool recorded = false;
        graph.addPass("gbuffer", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setColorTarget(0, color0);
                b.setColorTarget(1, color1);
                b.setDepthTarget(depth);
            },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.hasRenderTargets());
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                recorded = true;
            });

        // Consumer reads all three.
        bool readRan = false;
        graph.addPass("lighting", PassType::Graphics,
            [&](PassBuilder& b) {
                b.sampleImage(color0);
                b.sampleImage(color1);
                b.sampleImage(depth);
            },
            [&](PassContext&, VkCommandBuffer) { readRan = true; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded && readRan);
        std::printf("  Layer 1 multiple color targets: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        int order = 0;
        int aOrder = -1, bOrder = -1;

        graph.addPass("writeFirst", PassType::Graphics,
            [&](PassBuilder& b) { b.setColorTarget(0, img); },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                aOrder = order++;
            });

        graph.addPass("loadSecond", PassType::Graphics,
            [&](PassBuilder& b) {
                b.setColorTarget(0, img, LoadOp::Load);
            },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                bOrder = order++;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aOrder == 0 && "writer must execute first");
        assert(bOrder == 1 && "loader must execute second");
        std::printf("  Layer 1 LoadOp::Load ordering: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{};
        desc.width = 32; desc.height = 32;
        desc.format = VK_FORMAT_R8G8B8A8_UNORM;
        auto imgA = graph.createImage(desc);
        auto imgB = graph.createImage(desc);

        bool aRecorded = false, bRecorded = false;

        // Layer 0 pass: raw storage write.
        graph.addPass("compute", PassType::Compute,
            [&](PassBuilder& b) { b.writeStorageImage(imgA); },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(!ctx.hasRenderTargets());
                aRecorded = true;
            });

        // Layer 1 pass: render target.
        graph.addPass("render", PassType::Graphics,
            [&](PassBuilder& b) { b.setColorTarget(0, imgB); },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.hasRenderTargets());
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                bRecorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aRecorded && bRecorded);
        std::printf("  Layer 0 + Layer 1 coexistence: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{};
        desc.width = 32; desc.height = 32;
        desc.format = VK_FORMAT_R8G8B8A8_UNORM;
        auto transient = graph.createImage(desc);

        // An output to import so we have a consumer target.
        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto output = graph.importImage(testImage.value(), initState);

        int order = 0;
        int aOrder = -1, bOrder = -1;

        graph.addPass("renderToTransient", PassType::Graphics,
            [&](PassBuilder& b) { b.setColorTarget(0, transient); },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.vkImage(transient) != VK_NULL_HANDLE);
                assert(ctx.vkImageView(transient) != VK_NULL_HANDLE);
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                aOrder = order++;
            });

        graph.addPass("sampleTransient", PassType::Graphics,
            [&](PassBuilder& b) {
                b.sampleImage(transient);
                b.writeColorAttachment(output);
            },
            [&](PassContext&, VkCommandBuffer) { bOrder = order++; });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(aOrder == 0 && "render pass must execute first");
        assert(bOrder == 1 && "sample pass must execute second");
        std::printf("  Layer 1 transient with render targets: ok\n");
    }


    // Helper: create a simple pipeline layout (empty).
    auto makeEmptyLayout = [&]() {
        VkPipelineLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(vkDev, &ci, nullptr, &layout);
        return layout;
    };

    // Helper: create a DSL + pipeline layout for a given reflection.
    auto makeLayoutFromRefl = [&](const vksdl::ReflectedLayout& refl) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (const auto& rb : refl.bindings) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding         = rb.binding;
            lb.descriptorType  = rb.type;
            lb.descriptorCount = rb.count;
            lb.stageFlags      = rb.stages;
            bindings.push_back(lb);
        }
        VkDescriptorSetLayoutCreateInfo dslCI{};
        dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslCI.bindingCount = static_cast<std::uint32_t>(bindings.size());
        dslCI.pBindings    = bindings.data();
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(vkDev, &dslCI, nullptr, &dsl);

        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount = 1;
        plCI.pSetLayouts    = &dsl;
        VkPipelineLayout pl = VK_NULL_HANDLE;
        vkCreatePipelineLayout(vkDev, &plCI, nullptr, &pl);

        return std::pair{dsl, pl};
    };

    // Create a sampler for descriptor writing tests.
    auto testSampler = vksdl::SamplerBuilder(device.value())
        .linear()
        .clampToEdge()
        .build();
    assert(testSampler.ok());

    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto img = graph.importImage(testImage.value(), initState);

        // One COMBINED_IMAGE_SAMPLER binding named "tex".
        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "tex"});

        auto [dsl, pl] = makeLayoutFromRefl(refl);

        // Need a write target so the graph has something to order.
        auto output = graph.createImage({32, 32, VK_FORMAT_R8G8B8A8_UNORM});

        bool recorded = false;
        graph.addPass("layer2basic", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, output);
                b.setSampler(testSampler.value().vkSampler());
                b.bind("tex", img);
            },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                assert(ctx.hasPipeline());
                assert(ctx.hasRenderTargets());
                // Don't call beginRendering since pipeline is VK_NULL_HANDLE.
                // Just verify the descriptor set was allocated.
                assert(ctx.descriptorSet(0) != VK_NULL_HANDLE);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 basic bind: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto img0 = graph.createImage(desc, "a");
        auto img1 = graph.createImage(desc, "b");
        auto img2 = graph.createImage(desc, "c");
        auto img3 = graph.createImage(desc, "d");
        auto output = graph.createImage(desc, "out");

        // 4 writers.
        for (auto h : {img0, img1, img2, img3}) {
            auto cur = h;
            graph.addPass("write", PassType::Graphics,
                [cur](PassBuilder& b) { b.writeColorAttachment(cur); },
                [](PassContext&, VkCommandBuffer) {});
        }

        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "shadowDepth"});
        refl.bindings.push_back({0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "gbufAlbedo"});
        refl.bindings.push_back({0, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "gbufNormals"});
        refl.bindings.push_back({0, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "gbufDepth"});

        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool recorded = false;
        graph.addPass("lighting", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, output);
                b.setSampler(testSampler.value().vkSampler());
                b.bind("shadowDepth", img0);
                b.bind("gbufAlbedo",  img1);
                b.bind("gbufNormals", img2);
                b.bind("gbufDepth",   img3);
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.descriptorSet(0) != VK_NULL_HANDLE);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 multi-binding: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto imgA = graph.createImage(desc);
        auto imgB = graph.createImage(desc);

        // Layer 1 pass: writes imgA.
        bool layer1Ran = false;
        graph.addPass("layer1", PassType::Graphics,
            [&](PassBuilder& b) { b.setColorTarget(0, imgA); },
            [&](PassContext& ctx, VkCommandBuffer cmd) {
                ctx.beginRendering(cmd);
                ctx.endRendering(cmd);
                assert(!ctx.hasPipeline()); // Layer 1 only, no pipeline.
                layer1Ran = true;
            });

        // Layer 2 pass: reads imgA, writes imgB.
        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "input"});
        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool layer2Ran = false;
        graph.addPass("layer2", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, imgB);
                b.setSampler(testSampler.value().vkSampler());
                b.bind("input", imgA);
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.hasPipeline());
                assert(ctx.descriptorSet(0) != VK_NULL_HANDLE);
                layer2Ran = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(layer1Ran && layer2Ran);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 + Layer 1 coexistence: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto imgA = graph.createImage(desc);
        auto imgB = graph.createImage(desc);
        auto output = graph.createImage(desc);

        for (auto h : {imgA, imgB}) {
            auto cur = h;
            graph.addPass("write", PassType::Graphics,
                [cur](PassBuilder& b) { b.writeColorAttachment(cur); },
                [](PassContext&, VkCommandBuffer) {});
        }

        auto overrideSampler = vksdl::SamplerBuilder(device.value())
            .nearest()
            .clampToEdge()
            .build();
        assert(overrideSampler.ok());

        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "texA"});
        refl.bindings.push_back({0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "texB"});
        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool recorded = false;
        graph.addPass("sampler_test", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, output);
                b.setSampler(testSampler.value().vkSampler());
                b.bind("texA", imgA); // uses default sampler
                b.bind("texB", imgB, overrideSampler.value().vkSampler()); // override
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.descriptorSet(0) != VK_NULL_HANDLE);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 sampler override: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        BufferDesc bufDesc{256, 0};
        auto ubo = graph.createBuffer(bufDesc, "ubo");

        ImageDesc imgDesc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto output = graph.createImage(imgDesc);

        // Writer for the buffer.
        graph.addPass("fill_ubo", PassType::Compute,
            [&](PassBuilder& b) { b.writeStorageBuffer(ubo); },
            [](PassContext&, VkCommandBuffer) {});

        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "params"});
        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool recorded = false;
        graph.addPass("ubo_consumer", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, output);
                b.bind("params", ubo);
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.descriptorSet(0) != VK_NULL_HANDLE);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 uniform buffer binding: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto imgA = graph.createImage(desc);
        auto output = graph.createImage(desc);

        graph.addPass("write", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(imgA); },
            [](PassContext&, VkCommandBuffer) {});

        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "tex"});
        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool recorded = false;
        graph.addPass("auto_bind", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, output);
                b.setSampler(testSampler.value().vkSampler());
                b.bind("tex", imgA);
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                // Verify hasPipeline reports correctly.
                assert(ctx.hasPipeline());
                assert(ctx.hasRenderTargets());
                // We skip beginRendering here because VK_NULL_HANDLE pipeline
                // would cause a crash in vkCmdBindPipeline. The mechanism is
                // tested via the deferred example with real pipelines.
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 auto-bind integration: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto input  = graph.createImage(desc);
        auto storage = graph.createImage(desc);

        graph.addPass("writer", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(input); },
            [](PassContext&, VkCommandBuffer) {});

        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "tex"});

        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool recorded = false;
        graph.addPass("mixed", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setSampler(testSampler.value().vkSampler());
                b.bind("tex", input);
                // Layer 0 escape hatch: manual storage image write.
                b.writeStorageImage(storage);
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                assert(ctx.hasPipeline());
                assert(ctx.descriptorSet(0) != VK_NULL_HANDLE);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 0 + Layer 2 coexistence: ok\n");
    }

    {
        RenderGraph graph(device.value(), allocator.value());

        ImageDesc desc{32, 32, VK_FORMAT_R8G8B8A8_UNORM};
        auto img = graph.createImage(desc);
        auto output = graph.createImage(desc);

        graph.addPass("writer", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [](PassContext&, VkCommandBuffer) {});

        // 2 bindings, but only "managed" is in the bind map.
        // "external" is not bound -- user must write it manually.
        vksdl::ReflectedLayout refl;
        refl.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "managed"});
        refl.bindings.push_back({0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  1, VK_SHADER_STAGE_FRAGMENT_BIT, "external"});

        auto [dsl, pl] = makeLayoutFromRefl(refl);

        bool recorded = false;
        graph.addPass("partial", PassType::Graphics,
            VK_NULL_HANDLE, pl, refl,
            [&](PassBuilder& b) {
                b.setColorTarget(0, output);
                b.setSampler(testSampler.value().vkSampler());
                b.bind("managed", img);
                // "external" deliberately NOT bound.
            },
            [&](PassContext& ctx, VkCommandBuffer) {
                // Set is allocated (mixed: one managed, one external).
                VkDescriptorSet set = ctx.descriptorSet(0);
                assert(set != VK_NULL_HANDLE);
                recorded = true;
            });

        auto r = graph.compile();
        assert(r.ok());

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        graph.execute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(recorded);
        vkDestroyPipelineLayout(vkDev, pl, nullptr);
        vkDestroyDescriptorSetLayout(vkDev, dsl, nullptr);
        std::printf("  Layer 2 external descriptor: ok\n");
    }

    // prewarm
    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto img = graph.importImage(testImage.value(), initState);

        graph.addPass("draw", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) {});

        auto r = graph.prewarm();
        assert(r.ok());
        assert(!graph.isCompiled());

        graph.importImage(testImage.value(), initState);
        auto img2 = graph.importImage(testImage.value(), initState);

        graph.addPass("draw2", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img2); },
            [&](PassContext&, VkCommandBuffer) {});

        auto r2 = graph.compile();
        assert(r2.ok());
        assert(graph.isCompiled());

        std::printf("  prewarm: ok\n");
    }

    // compileAndExecute
    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto img = graph.importImage(testImage.value(), initState);

        bool recorded = false;
        graph.addPass("draw", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) { recorded = true; });

        auto oneShot = OneShotCmd::begin(vkDev, queueFamily);
        auto r = graph.compileAndExecute(oneShot.cmd);
        oneShot.submitAndWait(queue);

        assert(r.ok());
        assert(recorded);
        std::printf("  compileAndExecute: ok\n");
    }

    // stats
    {
        RenderGraph graph(device.value(), allocator.value());

        ResourceState initState{};
        initState.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        auto img = graph.importImage(testImage.value(), initState);

        graph.addPass("draw", PassType::Graphics,
            [&](PassBuilder& b) { b.writeColorAttachment(img); },
            [&](PassContext&, VkCommandBuffer) {});

        auto r = graph.compile();
        assert(r.ok());

        const auto& s = graph.stats();
        assert(s.passCount == 1);
        assert(s.imageBarrierCount >= 1);
        assert(s.compileTimeUs > 0.0);
        std::printf("  stats: ok\n");
    }

    std::printf("render graph test passed\n");
    return 0;
}
