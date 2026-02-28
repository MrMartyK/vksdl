// Demonstrates extended dynamic state (Vulkan 1.3 core).
// One pipeline handles all combinations of cull mode, front face,
// topology, and depth test. Keyboard controls:
//   1/2  cycle cull mode (none / back / front / front+back)
//   3/4  toggle front face (CCW / CW)
//   5/6  cycle topology (triangle list / line list / point list)
//   7/8  toggle depth test (on / off)
//   ESC  quit

#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_scancode.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

static vksdl::Mat4 mat4RotateY(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    vksdl::Mat4 out = vksdl::mat4Identity();
    out.at(0, 0) = c;
    out.at(2, 0) = s;
    out.at(0, 2) = -s;
    out.at(2, 2) = c;
    return out;
}

static vksdl::Mat4 mat4RotateX(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    vksdl::Mat4 out = vksdl::mat4Identity();
    out.at(1, 1) = c;
    out.at(2, 1) = s;
    out.at(1, 2) = -s;
    out.at(2, 2) = c;
    return out;
}

struct Vertex {
    float pos[3];
    float color[3];
};

static constexpr Vertex cubeVerts[] = {
    // Front face (red)
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.2f, 0.2f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.2f, 0.2f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.2f, 0.2f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.2f, 0.2f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.2f, 0.2f}},
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.2f, 0.2f}},
    // Back face (green)
    {{-0.5f, -0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
    {{0.5f, 0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
    {{0.5f, -0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
    {{0.5f, 0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
    {{-0.5f, -0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
    {{-0.5f, 0.5f, -0.5f}, {0.2f, 1.0f, 0.2f}},
    // Top face (blue)
    {{-0.5f, 0.5f, -0.5f}, {0.2f, 0.2f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.2f, 0.2f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.2f, 0.2f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.2f, 0.2f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.2f, 0.2f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.2f, 0.2f, 1.0f}},
    // Bottom face (yellow)
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.2f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.2f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.2f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.2f}},
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.2f}},
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.2f}},
    // Right face (magenta)
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.2f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.2f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.2f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.2f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.2f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.2f, 1.0f}},
    // Left face (cyan)
    {{-0.5f, -0.5f, -0.5f}, {0.2f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.2f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.2f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.2f, 1.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.2f, 1.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.2f, 1.0f, 1.0f}},
};
static constexpr std::uint32_t kVertexCount =
    static_cast<std::uint32_t>(sizeof(cubeVerts) / sizeof(cubeVerts[0]));

struct Push {
    float mvp[16];
};

static const char* cullName(VkCullModeFlags m) {
    switch (m) {
    case VK_CULL_MODE_NONE:
        return "none";
    case VK_CULL_MODE_BACK_BIT:
        return "back";
    case VK_CULL_MODE_FRONT_BIT:
        return "front";
    case VK_CULL_MODE_FRONT_AND_BACK:
        return "front+back";
    default:
        return "?";
    }
}

static const char* faceName(VkFrontFace f) {
    return (f == VK_FRONT_FACE_COUNTER_CLOCKWISE) ? "CCW" : "CW";
}

static const char* topoName(VkPrimitiveTopology t) {
    switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
        return "triangles";
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
        return "lines";
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
        return "points";
    default:
        return "?";
    }
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("Dynamic State", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_dynamic_state")
                        .requireVulkan(1, 3)
                        .enableWindowSupport()
                        .build()
                        .value();

    auto surface = vksdl::Surface::create(instance, window).value();

    auto device = vksdl::DeviceBuilder(instance, surface)
                      .needSwapchain()
                      .needDynamicRendering()
                      .needSync2()
                      .preferDiscreteGpu()
                      .build()
                      .value();

    auto swapchain =
        vksdl::SwapchainBuilder(device, surface).size(window.pixelSize()).build().value();

    auto frames = vksdl::FrameSync::create(device, swapchain.imageCount()).value();
    auto allocator = vksdl::Allocator::create(instance, device).value();

    auto depthImage = vksdl::ImageBuilder(allocator)
                          .size(swapchain.extent().width, swapchain.extent().height)
                          .depthAttachment()
                          .build()
                          .value();

    auto vertexBuffer =
        vksdl::uploadVertexBuffer(allocator, device, cubeVerts, sizeof(cubeVerts)).value();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
                        .vertexShader(shaderDir / "dynstate.vert.spv")
                        .fragmentShader(shaderDir / "dynstate.frag.spv")
                        .colorFormat(swapchain)
                        .depthFormat(VK_FORMAT_D32_SFLOAT)
                        .vertexBinding(0, sizeof(Vertex))
                        .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
                        .vertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                        .pushConstants<Push>(VK_SHADER_STAGE_VERTEX_BIT)
                        .dynamicCullMode()
                        .dynamicTopology()
                        .dynamicFrontFace()
                        .dynamicDepthTest()
                        .build()
                        .value();

    static const VkCullModeFlags kCullModes[] = {
        VK_CULL_MODE_NONE,
        VK_CULL_MODE_BACK_BIT,
        VK_CULL_MODE_FRONT_BIT,
        VK_CULL_MODE_FRONT_AND_BACK,
    };
    static const VkPrimitiveTopology kTopologies[] = {
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    };

    int cullIdx = 1; // back culling by default
    int topoIdx = 0; // triangle list by default
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depthTest = true;

    // Helper to update the window title with current state.
    auto updateTitle = [&]() {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "Dynamic State | Cull: %s | Face: %s | Topo: %s | Depth: %s | 1 pipeline",
                      cullName(kCullModes[cullIdx]), faceName(frontFace),
                      topoName(kTopologies[topoIdx]), depthTest ? "on" : "off");
        SDL_SetWindowTitle(window.sdlWindow(), buf);
    };
    updateTitle();

    std::printf(
        "Controls: 1/2 cull mode, 3/4 front face, 5/6 topology, 7/8 depth test, ESC quit\n");
    std::printf("Initial state: cull=%s face=%s topo=%s depth=%s\n", cullName(kCullModes[cullIdx]),
                faceName(frontFace), topoName(kTopologies[topoIdx]), depthTest ? "on" : "off");
    int frameNum = 0;

    bool running = true;
    vksdl::Event event;
    auto startTime = std::chrono::steady_clock::now();

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
                    cullIdx = (cullIdx + 1) % 4;
                    updateTitle();
                } else if (event.key == SDL_SCANCODE_2) {
                    cullIdx = (cullIdx + 3) % 4;
                    updateTitle();
                } else if (event.key == SDL_SCANCODE_3 || event.key == SDL_SCANCODE_4) {
                    frontFace = (frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE)
                                    ? VK_FRONT_FACE_CLOCKWISE
                                    : VK_FRONT_FACE_COUNTER_CLOCKWISE;
                    updateTitle();
                } else if (event.key == SDL_SCANCODE_5) {
                    topoIdx = (topoIdx + 1) % 3;
                    updateTitle();
                } else if (event.key == SDL_SCANCODE_6) {
                    topoIdx = (topoIdx + 2) % 3;
                    updateTitle();
                } else if (event.key == SDL_SCANCODE_7 || event.key == SDL_SCANCODE_8) {
                    depthTest = !depthTest;
                    updateTitle();
                }
            }
        }

        if (window.consumeResize()) {
            (void) swapchain.recreate(device, window);
            depthImage = vksdl::ImageBuilder(allocator)
                             .size(swapchain.extent().width, swapchain.extent().height)
                             .depthAttachment()
                             .build()
                             .value();
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        auto now = std::chrono::steady_clock::now();
        float t = std::chrono::duration<float>(now - startTime).count();
        float aspect = static_cast<float>(swapchain.extent().width) /
                       static_cast<float>(swapchain.extent().height);

        // Spin around Y, slight tilt around X for visual depth.
        vksdl::Mat4 model = mat4Mul(mat4RotateY(t * 0.8f), mat4RotateX(0.4f));

        float eye[3] = {0.0f, 1.5f, 3.0f};
        float target[3] = {0.0f, 0.0f, 0.0f};
        float up[3] = {0.0f, 1.0f, 0.0f};
        vksdl::Mat4 view = vksdl::lookAt(eye, target, up);

        // Forward-Z: near->0, far->1. Depth clear = 1.0, compare = LESS_OR_EQUAL.
        static constexpr float kFovY = 0.7854f; // 45 degrees in radians
        vksdl::Mat4 proj = vksdl::perspectiveForwardZ(kFovY, aspect, 0.1f, 100.0f);

        vksdl::Mat4 mvp = mat4Mul(proj, mat4Mul(view, model));

        Push push{};
        std::memcpy(push.mvp, mvp.data(), sizeof(push.mvp));

        vksdl::beginOneTimeCommands(frame.cmd);

        vksdl::transitionToColorAttachment(frame.cmd, img.image);
        vksdl::transitionToDepthAttachment(frame.cmd, depthImage.vkImage());

        VkRenderingAttachmentInfo colorAttach{};
        colorAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttach.imageView = img.view;
        colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttach.clearValue.color = {{0.08f, 0.08f, 0.10f, 1.0f}};

        VkRenderingAttachmentInfo depthAttach{};
        depthAttach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttach.imageView = depthImage.vkImageView();
        depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttach.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttach.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo renderInfo{};
        renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea = {{0, 0}, swapchain.extent()};
        renderInfo.layerCount = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments = &colorAttach;
        renderInfo.pDepthAttachment = &depthAttach;

        vkCmdBeginRendering(frame.cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain.extent().width);
        viewport.height = static_cast<float>(swapchain.extent().height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapchain.extent()};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        pipeline.bind(frame.cmd);
        pipeline.pushConstants(frame.cmd, push);

        vksdl::Pipeline::setCullMode(frame.cmd, kCullModes[cullIdx]);
        vksdl::Pipeline::setFrontFace(frame.cmd, frontFace);
        vksdl::Pipeline::setTopology(frame.cmd, kTopologies[topoIdx]);
        vksdl::Pipeline::setDepthTest(frame.cmd, depthTest);

        VkDeviceSize offset = 0;
        VkBuffer vb = vertexBuffer.vkBuffer();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vb, &offset);
        vkCmdDraw(frame.cmd, kVertexCount, 1, 0, 0);

        vkCmdEndRendering(frame.cmd);

        vksdl::transitionToPresent(frame.cmd, img.image);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        ++frameNum;
        if (frameNum % 60 == 0) {
            std::printf("[frame %d] cull=%s face=%s topo=%s depth=%s  (1 pipeline, 0 rebuilds)\n",
                        frameNum, cullName(kCullModes[cullIdx]), faceName(frontFace),
                        topoName(kTopologies[topoIdx]), depthTest ? "on" : "off");
        }
    }

    device.waitIdle();

    return 0;
}
