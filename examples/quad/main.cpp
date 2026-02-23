#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>

struct Vertex {
    float pos[2];
    float color[3];
};

static constexpr std::array<Vertex, 4> vertices = {{
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  // top-left: red
    {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},  // top-right: green
    {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},  // bottom-right: blue
    {{-0.5f,  0.5f}, {1.0f, 1.0f, 0.0f}},  // bottom-left: yellow
}};

static constexpr std::array<std::uint16_t, 6> indices = {
    0, 1, 2,  // first triangle
    2, 3, 0,  // second triangle
};

static void recordQuad(VkCommandBuffer cmd, VkExtent2D extent,
                       VkImage swapImage, VkImageView swapView,
                       const vksdl::Pipeline& pipeline,
                       VkBuffer vertexBuf, VkBuffer indexBuf) {
    vksdl::transitionToColorAttachment(cmd, swapImage);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView   = swapView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};

    VkRenderingInfo renderInfo{};
    renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea           = {{0, 0}, extent};
    renderInfo.layerCount           = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    pipeline.bind(cmd);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRendering(cmd);

    vksdl::transitionToPresent(cmd, swapImage);
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Quad", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_quad")
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

    auto vertexBuffer = vksdl::uploadVertexBuffer(allocator, device,
        vertices.data(), sizeof(Vertex) * vertices.size()).value();

    auto indexBuffer = vksdl::uploadIndexBuffer(allocator, device,
        indices.data(), sizeof(std::uint16_t) * indices.size()).value();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "quad.vert.spv")
        .fragmentShader(shaderDir / "quad.frag.spv")
        .colorFormat(swapchain)
        .vertexBinding(0, sizeof(Vertex))
        .vertexAttribute(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, pos))
        .vertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
        .build().value();

    bool running = true;
    vksdl::Event event;

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

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        vksdl::beginOneTimeCommands(frame.cmd);

        recordQuad(frame.cmd, swapchain.extent(),
                   img.image, img.view,
                   pipeline,
                   vertexBuffer.vkBuffer(), indexBuffer.vkBuffer());

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }

    device.waitIdle();

    return 0;
}
