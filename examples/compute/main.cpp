#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Compute Gradient", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_compute")
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

    auto pipeline = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "gradient.comp.spv")
        .descriptorSetLayout(descriptors.vkDescriptorSetLayout())
        .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

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

        auto now = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        VkCommandBuffer cmd = frame.cmd;

        vksdl::beginOneTimeCommands(cmd);

        vksdl::transitionToComputeWrite(cmd, storageImage.vkImage());

        pipeline.bind(cmd, descriptors);
        pipeline.pushConstants(cmd, time);

        std::uint32_t groupsX = (swapchain.extent().width  + 15) / 16;
        std::uint32_t groupsY = (swapchain.extent().height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vksdl::blitToSwapchain(cmd, storageImage,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               img.image, swapchain.extent());

        vksdl::endCommands(cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    device.waitIdle();

    return 0;
}
