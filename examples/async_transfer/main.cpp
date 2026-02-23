#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <vector>

// Demonstrates TransferQueue streaming: new colored triangles are uploaded to
// the GPU asynchronously every ~30 frames. Each appears around a ring once its
// transfer completes. Ownership acquire barriers are inserted in the graphics
// command buffer for cross-family transfers.
//
// Window title: Async Transfer | Dedicated: yes/no | Shapes: N/24 ready | Pending: P

static constexpr int MAX_SHAPES       = 24;
static constexpr int FRAMES_PER_SHAPE = 30; // upload a new shape every N frames

struct Vertex {
    float pos[2];
    float color[3];
};

// Per-shape state. Buffer is created and filled via uploadAsync.
// pending is valid until barrierDone becomes true.
struct Shape {
    std::optional<vksdl::Buffer>          vertexBuffer;
    std::optional<vksdl::PendingTransfer> pending;
    bool                                  ready       = false; // transfer complete
    bool                                  barrierDone = false; // acquire barrier inserted
};

static void hsvToRgb(float h, float s, float v,
                     float& r, float& g, float& b)
{
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    if      (h < 1.0f / 6.0f) { r = c + m; g = x + m; b = m;     }
    else if (h < 2.0f / 6.0f) { r = x + m; g = c + m; b = m;     }
    else if (h < 3.0f / 6.0f) { r = m;     g = c + m; b = x + m; }
    else if (h < 4.0f / 6.0f) { r = m;     g = x + m; b = c + m; }
    else if (h < 5.0f / 6.0f) { r = x + m; g = m;     b = c + m; }
    else                       { r = c + m; g = m;     b = x + m; }
}

// Generate three vertices for a small equilateral triangle centered at (cx, cy).
static void makeTriangle(int shapeIndex, Vertex out[3])
{
    float angle = static_cast<float>(shapeIndex) * (2.0f * std::numbers::pi_v<float> / MAX_SHAPES);
    float cx = 0.72f * std::cos(angle);
    float cy = 0.72f * std::sin(angle);
    float r  = 0.085f;

    float hue = static_cast<float>(shapeIndex) / static_cast<float>(MAX_SHAPES);
    float red, green, blue;
    hsvToRgb(hue, 0.85f, 1.0f, red, green, blue);

    // Three vertices evenly spaced around the center, pointing outward.
    for (int i = 0; i < 3; ++i) {
        float a = angle + static_cast<float>(i) * (2.0f * std::numbers::pi_v<float> / 3.0f);
        out[i].pos[0]   = cx + r * std::cos(a);
        out[i].pos[1]   = cy + r * std::sin(a);
        out[i].color[0] = red;
        out[i].color[1] = green;
        out[i].color[2] = blue;
    }
}

int main() {
    auto app    = vksdl::App::create().value();
    auto window = app.createWindow("Async Transfer", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_async_transfer")
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
    auto transferQ = vksdl::TransferQueue::create(device, allocator).value();

    std::printf("GPU: %s\n", device.gpuName());
    std::printf("Dedicated transfer queue: %s\n",
                device.hasDedicatedTransfer() ? "yes" : "no (using graphics)");
    std::printf("Transfer family: %u  Graphics family: %u\n",
                device.queueFamilies().transfer,
                device.queueFamilies().graphics);

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "transfer_demo.vert.spv")
        .fragmentShader(shaderDir / "transfer_demo.frag.spv")
        .colorFormat(swapchain)
        .vertexBinding(0, sizeof(Vertex))
        .vertexAttribute(0, 0, VK_FORMAT_R32G32_SFLOAT,
                         static_cast<std::uint32_t>(offsetof(Vertex, pos)))
        .vertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                         static_cast<std::uint32_t>(offsetof(Vertex, color)))
        .pushConstants<float>(VK_SHADER_STAGE_VERTEX_BIT)
        .build().value();

    // Shape ring: slots are filled one at a time via async transfers.
    Shape shapes[MAX_SHAPES];

    bool     running    = true;
    int      frameCount = 0;
    int      nextShape  = 0; // next slot to upload
    vksdl::Event event;
    auto startTime = std::chrono::steady_clock::now();

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);
        }

        if (nextShape < MAX_SHAPES && (frameCount % FRAMES_PER_SHAPE) == 0) {
            Vertex verts[3];
            makeTriangle(nextShape, verts);

            auto buf = vksdl::BufferBuilder(allocator)
                .vertexBuffer()
                .size(sizeof(verts))
                .build();

            if (buf.ok()) {
                shapes[nextShape].vertexBuffer = std::move(buf.value());

                auto pend = transferQ.uploadAsync(
                    *shapes[nextShape].vertexBuffer, verts, sizeof(verts));

                if (pend.ok()) {
                    shapes[nextShape].pending = pend.value();
                    std::printf("Shape %2d: upload started (timeline=%llu, xfer=%s)\n",
                                nextShape,
                                static_cast<unsigned long long>(pend.value().timelineValue),
                                pend.value().needsOwnershipTransfer ? "cross-family" : "same-family");
                } else {
                    std::printf("Shape %d: uploadAsync failed: %s\n",
                                nextShape, pend.error().message.c_str());
                    shapes[nextShape].vertexBuffer.reset();
                }
            } else {
                std::printf("Shape %d: buffer alloc failed: %s\n",
                            nextShape, buf.error().message.c_str());
            }

            ++nextShape;
        }

        int pendingCount = 0;
        for (int i = 0; i < nextShape; ++i) {
            if (!shapes[i].pending.has_value()) continue;
            if (shapes[i].ready) continue;

            if (transferQ.isComplete(shapes[i].pending->timelineValue)) {
                shapes[i].ready = true;
                std::printf("Shape %2d: transfer complete, will acquire on next frame\n", i);
            } else {
                ++pendingCount;
            }
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        auto now  = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        vksdl::beginOneTimeCommands(frame.cmd);

        for (int i = 0; i < nextShape; ++i) {
            if (!shapes[i].ready)       continue;
            if (shapes[i].barrierDone)  continue;
            if (!shapes[i].pending.has_value()) continue;

            vksdl::TransferQueue::insertAcquireBarrier(frame.cmd, *shapes[i].pending);
            shapes[i].barrierDone = true;
        }

        vksdl::transitionToColorAttachment(frame.cmd, img.image);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView   = img.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.05f, 0.05f, 0.08f, 1.0f}};

        VkRenderingInfo renderInfo{};
        renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea           = {{0, 0}, swapchain.extent()};
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;

        vkCmdBeginRendering(frame.cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(swapchain.extent().width);
        viewport.height   = static_cast<float>(swapchain.extent().height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapchain.extent()};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        pipeline.bind(frame.cmd);
        pipeline.pushConstants(frame.cmd, time);

        // Draw all shapes whose acquire barrier has been recorded.
        for (int i = 0; i < nextShape; ++i) {
            if (!shapes[i].barrierDone) continue;
            if (!shapes[i].vertexBuffer.has_value()) continue;

            VkDeviceSize offset = 0;
            VkBuffer     vb     = shapes[i].vertexBuffer->vkBuffer();
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vb, &offset);
            vkCmdDraw(frame.cmd, 3, 1, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);

        vksdl::transitionToPresent(frame.cmd, img.image);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        if ((frameCount % 60) == 0) {
            int readyCount = 0;
            for (int i = 0; i < MAX_SHAPES; ++i) {
                if (shapes[i].ready) ++readyCount;
            }

            char title[256];
            std::snprintf(title, sizeof(title),
                          "Async Transfer | Dedicated: %s | Shapes: %d/%d ready | Pending: %d",
                          device.hasDedicatedTransfer() ? "yes" : "no",
                          readyCount, MAX_SHAPES, pendingCount);
            SDL_SetWindowTitle(window.sdlWindow(), title);
        }

        ++frameCount;
    }

    device.waitIdle();

    return 0;
}
