#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>

struct Vertex {
    float pos[3];
    float color[3];
};

struct MVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// 8 unique cube vertices with distinct colors.
static constexpr std::array<Vertex, 8> vertices = {{
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, // 0: red
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},  // 1: green
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},   // 2: blue
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},  // 3: yellow
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},  // 4: magenta
    {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},   // 5: cyan
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},    // 6: white
    {{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}},   // 7: gray
}};

// 36 indices for 12 triangles (6 faces).
static constexpr std::array<std::uint16_t, 36> indices = {
    // Front face  (z = -0.5)
    0,
    1,
    2,
    2,
    3,
    0,
    // Back face   (z = +0.5)
    5,
    4,
    7,
    7,
    6,
    5,
    // Top face    (y = +0.5)
    3,
    2,
    6,
    6,
    7,
    3,
    // Bottom face (y = -0.5)
    4,
    5,
    1,
    1,
    0,
    4,
    // Right face  (x = +0.5)
    1,
    5,
    6,
    6,
    2,
    1,
    // Left face   (x = -0.5)
    4,
    0,
    3,
    3,
    7,
    4,
};

static void recordCube(VkCommandBuffer cmd, VkExtent2D extent, VkImage swapImage,
                       VkImageView swapView, VkImage depthImage, VkImageView depthView,
                       const vksdl::Pipeline& pipeline, const vksdl::DescriptorSet& descriptorSet,
                       VkBuffer vertexBuf, VkBuffer indexBuf) {
    vksdl::transitionToColorAttachment(cmd, swapImage);
    vksdl::transitionToDepthAttachment(cmd, depthImage);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderInfo{};
    renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea = {{0, 0}, extent};
    renderInfo.layerCount = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments = &colorAttachment;
    renderInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    pipeline.bind(cmd, descriptorSet);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRendering(cmd);

    vksdl::transitionToPresent(cmd, swapImage);
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Spinning Cube", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_cube")
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

    auto vertexBuffer = vksdl::uploadVertexBuffer(allocator, device, vertices.data(),
                                                  sizeof(Vertex) * vertices.size())
                            .value();

    auto indexBuffer = vksdl::uploadIndexBuffer(allocator, device, indices.data(),
                                                sizeof(std::uint16_t) * indices.size())
                           .value();

    auto mvpBuffer =
        vksdl::BufferBuilder(allocator).size(sizeof(MVP)).uniformBuffer().build().value();

    auto descriptors = vksdl::DescriptorSetBuilder(device)
                           .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
                           .build()
                           .value();

    descriptors.updateBuffer(0, mvpBuffer.vkBuffer(), sizeof(MVP));

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
                        .vertexShader(shaderDir / "cube.vert.spv")
                        .fragmentShader(shaderDir / "cube.frag.spv")
                        .colorFormat(swapchain)
                        .depthFormat(VK_FORMAT_D32_SFLOAT)
                        .vertexBinding(0, sizeof(Vertex))
                        .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
                        .vertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
                        .descriptorSetLayout(descriptors.vkDescriptorSetLayout())
                        .cullBack()
                        .clockwise()
                        .build()
                        .value();

    bool running = true;
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
            (void) swapchain.recreate(device, window);
            depthImage = vksdl::ImageBuilder(allocator)
                             .size(swapchain.extent().width, swapchain.extent().height)
                             .depthAttachment()
                             .build()
                             .value();
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        auto now = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        float aspect = static_cast<float>(swapchain.extent().width) /
                       static_cast<float>(swapchain.extent().height);

        MVP mvp;
        mvp.model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0.0f, 1.0f, 0.0f));
        mvp.view = glm::lookAt(glm::vec3(0.0f, 1.5f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f));
        mvp.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        mvp.proj[1][1] *= -1.0f; // Vulkan Y is flipped vs OpenGL

        std::memcpy(mvpBuffer.mappedData(), &mvp, sizeof(MVP));

        vksdl::beginOneTimeCommands(frame.cmd);

        recordCube(frame.cmd, swapchain.extent(), img.image, img.view, depthImage.vkImage(),
                   depthImage.vkImageView(), pipeline, descriptors, vertexBuffer.vkBuffer(),
                   indexBuffer.vkBuffer());

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    device.waitIdle();

    return 0;
}
