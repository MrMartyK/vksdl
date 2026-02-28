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
    float uv[2];
};

struct MVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// 24 vertices (4 per face) for correct UV mapping.
// Each face maps the full texture (0,0) to (1,1).
static constexpr std::array<Vertex, 24> vertices = {{
    // Front face (z = -0.5, facing -Z)
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}},

    // Back face (z = +0.5, facing +Z)
    {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},

    // Top face (y = +0.5, facing +Y)
    {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},

    // Bottom face (y = -0.5, facing -Y)
    {{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f}},

    // Right face (x = +0.5, facing +X)
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f}},

    // Left face (x = -0.5, facing -X)
    {{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},
}};

// 36 indices for 12 triangles (6 faces, 2 triangles each).
static constexpr std::array<std::uint16_t, 36> indices = {
    0,  1,  2,  2,  3,  0,  // front
    4,  5,  6,  6,  7,  4,  // back
    8,  9,  10, 10, 11, 8,  // top
    12, 13, 14, 14, 15, 12, // bottom
    16, 17, 18, 18, 19, 16, // right
    20, 21, 22, 22, 23, 20, // left
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
    auto window = app.createWindow("vksdl - Textured Cube", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_textured_cube")
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

    std::filesystem::path textureDir = vksdl::exeDir() / "textures";

    auto imageData = vksdl::loadImage(textureDir / "checkerboard.png").value();

    auto textureImage = vksdl::ImageBuilder(allocator)
                            .size(imageData.width, imageData.height)
                            .format(VK_FORMAT_R8G8B8A8_SRGB)
                            .sampled()
                            .build()
                            .value();

    if (!vksdl::uploadToImage(allocator, device, textureImage, imageData.pixels,
                              imageData.sizeBytes())
             .ok())
        return 1;

    auto sampler = vksdl::SamplerBuilder(device).nearest().repeat().build().value();

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
                           .addCombinedImageSampler(1, VK_SHADER_STAGE_FRAGMENT_BIT)
                           .build()
                           .value();

    descriptors.updateBuffer(0, mvpBuffer.vkBuffer(), sizeof(MVP));
    descriptors.updateImage(1, textureImage.vkImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            sampler.vkSampler());

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
                        .vertexShader(shaderDir / "textured_cube.vert.spv")
                        .fragmentShader(shaderDir / "textured_cube.frag.spv")
                        .colorFormat(swapchain)
                        .depthFormat(VK_FORMAT_D32_SFLOAT)
                        .vertexBinding(0, sizeof(Vertex))
                        .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
                        .vertexAttribute(1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv))
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
