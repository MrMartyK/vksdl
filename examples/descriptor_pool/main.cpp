#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

// SDL escape hatch for SDL_SetWindowTitle.
#include <SDL3/SDL.h>

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
#include <string>
#include <vector>

// 100 spinning 3D cubes (10x10 grid). Each cube has its own UBO and descriptor
// set allocated from a shared DescriptorPool. Colors cycle the full hue wheel.

static constexpr int GRID_SIZE     = 10;
static constexpr int OBJECT_COUNT  = GRID_SIZE * GRID_SIZE;
static constexpr float CUBE_SPACING = 1.5f;

// std140: mat4 (64) + vec4 (16) = 80 bytes.
// The shader reads ubo.color.rgb; .w is unused padding.
struct ObjectUBO {
    glm::mat4 mvp;
    glm::vec4 color; // .w unused
};

// Cube: 36 positions-only vertices (6 faces * 2 triangles * 3 vertices).
// CCW winding when viewed from outside.
static constexpr float cubePositions[] = {
    // Front face (z = +0.5)
    -0.5f, -0.5f,  0.5f,   0.5f, -0.5f,  0.5f,   0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,  -0.5f,  0.5f,  0.5f,  -0.5f, -0.5f,  0.5f,
    // Back face (z = -0.5)
    -0.5f, -0.5f, -0.5f,   0.5f,  0.5f, -0.5f,   0.5f, -0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,  -0.5f, -0.5f, -0.5f,  -0.5f,  0.5f, -0.5f,
    // Top face (y = +0.5)
    -0.5f,  0.5f, -0.5f,   0.5f,  0.5f,  0.5f,   0.5f,  0.5f, -0.5f,
     0.5f,  0.5f,  0.5f,  -0.5f,  0.5f, -0.5f,  -0.5f,  0.5f,  0.5f,
    // Bottom face (y = -0.5)
    -0.5f, -0.5f, -0.5f,   0.5f, -0.5f, -0.5f,   0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,  -0.5f, -0.5f,  0.5f,  -0.5f, -0.5f, -0.5f,
    // Right face (x = +0.5)
     0.5f, -0.5f, -0.5f,   0.5f,  0.5f, -0.5f,   0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,   0.5f, -0.5f,  0.5f,   0.5f, -0.5f, -0.5f,
    // Left face (x = -0.5)
    -0.5f, -0.5f, -0.5f,  -0.5f,  0.5f,  0.5f,  -0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f,  0.5f,  -0.5f, -0.5f, -0.5f,  -0.5f, -0.5f,  0.5f,
};

// Convert HSV (h in [0,1), s/v in [0,1]) to linear RGB.
static glm::vec3 hsvToRgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    glm::vec3 rgb;
    if      (h < 1.0f / 6.0f) rgb = {c, x, 0.0f};
    else if (h < 2.0f / 6.0f) rgb = {x, c, 0.0f};
    else if (h < 3.0f / 6.0f) rgb = {0.0f, c, x};
    else if (h < 4.0f / 6.0f) rgb = {0.0f, x, c};
    else if (h < 5.0f / 6.0f) rgb = {x, 0.0f, c};
    else                       rgb = {c, 0.0f, x};
    return rgb + glm::vec3(m);
}

int main() {
    auto app    = vksdl::App::create().value();
    auto window = app.createWindow("Descriptor Pool", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_descriptor_pool")
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

    auto makeDepth = [&]() {
        return vksdl::ImageBuilder(allocator)
            .size(swapchain.extent().width, swapchain.extent().height)
            .depthAttachment()
            .build().value();
    };
    auto depthImage = makeDepth();

    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding         = 0;
    uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 1;
    layoutCI.pBindings    = &uboBinding;

    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorSetLayout(device.vkDevice(), &layoutCI, nullptr, &dsLayout);
    if (vr != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateDescriptorSetLayout failed: %d\n", vr);
        return 1;
    }

    auto pool = vksdl::DescriptorPool::create(device).value();

    struct ObjectData {
        vksdl::Buffer   ubo;
        VkDescriptorSet set;
        glm::vec3       color;
        float           spinSpeed; // radians per second
    };

    std::vector<ObjectData> objects;
    objects.reserve(OBJECT_COUNT);

    for (int i = 0; i < OBJECT_COUNT; ++i) {
        auto ubo = vksdl::BufferBuilder(allocator)
            .uniformBuffer()
            .size(sizeof(ObjectUBO))
            .mapped()
            .build().value();

        auto set = pool.allocate(dsLayout).value();

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = ubo.vkBuffer();
        bufInfo.offset = 0;
        bufInfo.range  = sizeof(ObjectUBO);

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = set;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo     = &bufInfo;

        vkUpdateDescriptorSets(device.vkDevice(), 1, &write, 0, nullptr);

        // Even hue distribution across the full wheel; full saturation and
        // high value so all cubes are vivid but not clipping.
        float hue = static_cast<float>(i) / static_cast<float>(OBJECT_COUNT);
        glm::vec3 color = hsvToRgb(hue, 0.85f, 0.95f);

        // Speed varies between ~0.5 and ~2.5 rad/s spread across the grid.
        float spinSpeed = 0.5f + 2.0f * (static_cast<float>(i) / (OBJECT_COUNT - 1));

        objects.push_back({std::move(ubo), set, color, spinSpeed});
    }

    std::printf("Pool: %u sets allocated across %u internal pool(s)\n",
                pool.allocatedSetCount(), pool.poolCount());

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "pool_cube.vert.spv")
        .fragmentShader(shaderDir / "pool_cube.frag.spv")
        .colorFormat(swapchain)
        .depthFormat(VK_FORMAT_D32_SFLOAT)
        .vertexBinding(0, sizeof(float) * 3)
        .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .descriptorSetLayout(dsLayout)
        .cullBack()
        .build().value();

    auto vertexBuffer = vksdl::uploadVertexBuffer(
        allocator, device, cubePositions, sizeof(cubePositions)).value();

    // Camera positioned to see the full 10x10 grid (9*1.5 = 13.5 units wide).
    float gridHalfExtent = (GRID_SIZE - 1) * CUBE_SPACING * 0.5f;
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, gridHalfExtent + 2.0f, gridHalfExtent + 14.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));

    auto startTime = std::chrono::steady_clock::now();
    bool running   = true;
    vksdl::Event event;
    int frameNum = 0;

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);
            depthImage = makeDepth();
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        float elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - startTime).count();

        float aspect = static_cast<float>(swapchain.extent().width)
                     / static_cast<float>(swapchain.extent().height);
        glm::mat4 proj = glm::perspective(glm::radians(50.0f), aspect, 0.1f, 200.0f);
        proj[1][1] *= -1.0f; // Vulkan clip-space Y flip

        for (int i = 0; i < OBJECT_COUNT; ++i) {
            int row = i / GRID_SIZE;
            int col = i % GRID_SIZE;

            float x = (col - (GRID_SIZE - 1) * 0.5f) * CUBE_SPACING;
            float y = 0.0f;
            float z = (row - (GRID_SIZE - 1) * 0.5f) * CUBE_SPACING;

            float angle = elapsed * objects[static_cast<std::size_t>(i)].spinSpeed;

            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
            model = glm::rotate(model, angle, glm::vec3(0.0f, 1.0f, 0.0f));

            ObjectUBO uboData;
            uboData.mvp   = proj * view * model;
            uboData.color = glm::vec4(objects[static_cast<std::size_t>(i)].color, 0.0f);

            std::memcpy(objects[static_cast<std::size_t>(i)].ubo.mappedData(),
                        &uboData, sizeof(ObjectUBO));
        }

        {
            std::string title = "Descriptor Pool | "
                + std::to_string(pool.allocatedSetCount()) + " sets across "
                + std::to_string(pool.poolCount()) + " pool(s) | "
                + std::to_string(OBJECT_COUNT) + " cubes";
            SDL_SetWindowTitle(window.sdlWindow(), title.c_str());
        }

        vksdl::beginOneTimeCommands(frame.cmd);

        vksdl::transitionToColorAttachment(frame.cmd, img.image);
        vksdl::transitionToDepthAttachment(frame.cmd, depthImage.vkImage());

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView   = img.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.05f, 0.05f, 0.08f, 1.0f}};

        VkRenderingAttachmentInfo depthAttachment{};
        depthAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView   = depthImage.vkImageView();
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo renderInfo{};
        renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea           = {{0, 0}, swapchain.extent()};
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;
        renderInfo.pDepthAttachment     = &depthAttachment;

        vkCmdBeginRendering(frame.cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(swapchain.extent().width);
        viewport.height   = static_cast<float>(swapchain.extent().height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapchain.extent()};
        vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline.vkPipeline());

        VkDeviceSize vertexOffset = 0;
        VkBuffer vb = vertexBuffer.vkBuffer();
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vb, &vertexOffset);

        static constexpr std::uint32_t CUBE_VERTEX_COUNT = 36;
        for (int i = 0; i < OBJECT_COUNT; ++i) {
            vkCmdBindDescriptorSets(frame.cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.vkPipelineLayout(),
                                    0, 1,
                                    &objects[static_cast<std::size_t>(i)].set,
                                    0, nullptr);
            vkCmdDraw(frame.cmd, CUBE_VERTEX_COUNT, 1, 0, 0);
        }

        vkCmdEndRendering(frame.cmd);

        vksdl::transitionToPresent(frame.cmd, img.image);

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        ++frameNum;
        if (frameNum % 60 == 0) {
            std::printf("[frame %d] %u sets across %u pool(s), %d cubes rendering\n",
                        frameNum, pool.allocatedSetCount(), pool.poolCount(),
                        OBJECT_COUNT);
        }
    }

    device.waitIdle();

    vkDestroyDescriptorSetLayout(device.vkDevice(), dsLayout, nullptr);

    return 0;
}
