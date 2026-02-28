#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numbers>

static constexpr int CUBE_COUNT = 5;
static constexpr float CIRCLE_RADIUS = 2.5f;

struct Vertex {
    float pos[3];
    float uv[2];
};

struct SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
};

struct ObjectUBO {
    glm::mat4 model;
};

// 24 vertices (4 per face) for correct UV mapping.
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

static constexpr std::array<std::uint16_t, 36> indices = {
    0,  1,  2,  2,  3,  0,  // front
    4,  5,  6,  6,  7,  4,  // back
    8,  9,  10, 10, 11, 8,  // top
    12, 13, 14, 14, 15, 12, // bottom
    16, 17, 18, 18, 19, 16, // right
    20, 21, 22, 22, 23, 20, // left
};

static void recordScene(VkCommandBuffer cmd, VkExtent2D extent, VkImage swapImage,
                        VkImageView swapView, VkImage depthImage, VkImageView depthView,
                        const vksdl::Pipeline& pipeline, VkDescriptorSet sceneSet,
                        VkDescriptorSet materialSet, VkBuffer vertexBuf, VkBuffer indexBuf,
                        VkDeviceSize dynamicStride) {
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

    pipeline.bind(cmd);

    // Bind material set (set 1) once -- all cubes share the same texture
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.vkPipelineLayout(), 1, 1,
                            &materialSet, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuf, 0, VK_INDEX_TYPE_UINT16);

    // Draw each cube with its dynamic UBO offset
    for (int i = 0; i < CUBE_COUNT; ++i) {
        std::uint32_t dynamicOffset = static_cast<std::uint32_t>(dynamicStride * i);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.vkPipelineLayout(),
                                0, 1, &sceneSet, 1, &dynamicOffset);
        vkCmdDrawIndexed(cmd, static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    vksdl::transitionToPresent(cmd, swapImage);
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Multi Object", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_multi_object")
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
                            .mipmapped()
                            .build()
                            .value();

    if (!vksdl::uploadToImage(allocator, device, textureImage, imageData.pixels,
                              imageData.sizeBytes())
             .ok())
        return 1;

    // Generate mipmaps (level 0 left in TRANSFER_DST by uploadToImage).
    {
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCI.queueFamilyIndex = device.queueFamilies().graphics;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(device.vkDevice(), &poolCI, nullptr, &cmdPool) != VK_SUCCESS)
            return 1;

        VkCommandBufferAllocateInfo cmdAI{};
        cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI.commandPool = cmdPool;
        cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device.vkDevice(), &cmdAI, &cmd) != VK_SUCCESS) {
            vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
            return 1;
        }

        vksdl::beginOneTimeCommands(cmd);
        vksdl::generateMipmaps(cmd, textureImage);
        vksdl::endCommands(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        if (vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
            return 1;
        }
        vkQueueWaitIdle(device.graphicsQueue());
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
    }

    auto sampler = vksdl::SamplerBuilder(device).linear().repeat().build().value();

    auto vertexBuffer = vksdl::uploadVertexBuffer(allocator, device, vertices.data(),
                                                  sizeof(Vertex) * vertices.size())
                            .value();

    auto indexBuffer = vksdl::uploadIndexBuffer(allocator, device, indices.data(),
                                                sizeof(std::uint16_t) * indices.size())
                           .value();

    auto sceneBuffer =
        vksdl::BufferBuilder(allocator).size(sizeof(SceneUBO)).uniformBuffer().build().value();

    VkDeviceSize dynamicStride =
        vksdl::alignUp(sizeof(ObjectUBO), device.minUniformBufferOffsetAlignment());
    VkDeviceSize dynamicBufSize = dynamicStride * CUBE_COUNT;

    auto objectBuffer =
        vksdl::BufferBuilder(allocator).size(dynamicBufSize).uniformBuffer().build().value();

    // Set 0: scene globals (view/proj) + per-object dynamic UBO (model)
    auto sceneSet = vksdl::DescriptorSetBuilder(device)
                        .addUniformBuffer(0, VK_SHADER_STAGE_VERTEX_BIT)
                        .addDynamicUniformBuffer(1, VK_SHADER_STAGE_VERTEX_BIT)
                        .build()
                        .value();

    sceneSet.updateBuffer(0, sceneBuffer.vkBuffer(), sizeof(SceneUBO));
    sceneSet.updateBuffer(1, objectBuffer.vkBuffer(), sizeof(ObjectUBO));

    // Set 1: material (shared texture)
    auto materialSet = vksdl::DescriptorSetBuilder(device)
                           .addCombinedImageSampler(0, VK_SHADER_STAGE_FRAGMENT_BIT)
                           .build()
                           .value();

    materialSet.updateImage(0, textureImage.vkImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            sampler.vkSampler());

    vksdl::debugName(device.vkDevice(), sceneBuffer.vkBuffer(), "scene UBO");
    vksdl::debugName(device.vkDevice(), objectBuffer.vkBuffer(), "object dynamic UBO");
    vksdl::debugName(device.vkDevice(), textureImage.vkImage(), "cube texture");
    vksdl::debugName(device.vkDevice(), sampler.vkSampler(), "trilinear sampler");

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
                        .vertexShader(shaderDir / "multi_object.vert.spv")
                        .fragmentShader(shaderDir / "multi_object.frag.spv")
                        .colorFormat(swapchain)
                        .depthFormat(VK_FORMAT_D32_SFLOAT)
                        .vertexBinding(0, sizeof(Vertex))
                        .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
                        .vertexAttribute(1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv))
                        .descriptorSetLayout(sceneSet.vkDescriptorSetLayout())
                        .descriptorSetLayout(materialSet.vkDescriptorSetLayout())
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

        SceneUBO scene;
        scene.view = glm::lookAt(glm::vec3(0.0f, 3.0f, 6.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
        scene.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        scene.proj[1][1] *= -1.0f;

        std::memcpy(sceneBuffer.mappedData(), &scene, sizeof(SceneUBO));

        auto* objectData = static_cast<char*>(objectBuffer.mappedData());
        for (int i = 0; i < CUBE_COUNT; ++i) {
            float angle = static_cast<float>(i) * 2.0f * std::numbers::pi_v<float> / CUBE_COUNT;
            float x = CIRCLE_RADIUS * std::cos(angle);
            float z = CIRCLE_RADIUS * std::sin(angle);

            float speed = 1.0f + static_cast<float>(i) * 0.5f;
            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.0f, z));
            model = glm::rotate(model, time * speed, glm::vec3(0.0f, 1.0f, 0.0f));

            auto* dst = reinterpret_cast<ObjectUBO*>(objectData + dynamicStride * i);
            dst->model = model;
        }

        vksdl::beginOneTimeCommands(frame.cmd);

        recordScene(frame.cmd, swapchain.extent(), img.image, img.view, depthImage.vkImage(),
                    depthImage.vkImageView(), pipeline, sceneSet.vkDescriptorSet(),
                    materialSet.vkDescriptorSet(), vertexBuffer.vkBuffer(), indexBuffer.vkBuffer(),
                    dynamicStride);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    device.waitIdle();

    return 0;
}
