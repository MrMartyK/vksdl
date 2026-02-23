#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>

// Demonstrates shader reflection in a two-pass compute pipeline:
//
//   Pass 1 (manual layout):    gen_scene.comp   -> image A
//   Pass 2 (reflected layout): edge_filter.comp -> image B
//
// The edge filter pipeline uses reflectDescriptors() -- no manual
// descriptorSetLayout() or pushConstants() calls. The SPIR-V metadata
// is the entire API contract. Image B is blitted to the swapchain.

// Push constant struct for the edge filter -- must match edge_filter.comp.
struct FilterPush {
    float time;
    float edgeStrength;
};

static vksdl::Result<vksdl::Image> makeStorageImage(
    const vksdl::Allocator& allocator, VkExtent2D extent)
{
    return vksdl::ImageBuilder(allocator)
        .size(extent.width, extent.height)
        .format(VK_FORMAT_R8G8B8A8_UNORM)
        .storage()
        .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .build();
}

// Write two storage image descriptors into a raw VkDescriptorSet.
static void writeFilterDescriptors(VkDevice device,
                                   VkDescriptorSet ds,
                                   VkImageView inputView,
                                   VkImageView outputView)
{
    VkDescriptorImageInfo imageInfos[2] = {};
    imageInfos[0].imageView   = inputView;
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[1].imageView   = outputView;
    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = ds;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo      = &imageInfos[0];

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = ds;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &imageInfos[1];

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

int main() {
    auto app    = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - Shader Reflection", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_shader_reflect")
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

    VkExtent2D extent = swapchain.extent();

    // Image A: gen_scene writes the animated source pattern.
    // Image B: edge_filter reads A and writes the cel-shaded output.
    auto imageA = makeStorageImage(allocator, extent).value();
    auto imageB = makeStorageImage(allocator, extent).value();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto genDesc = vksdl::DescriptorSetBuilder(device)
        .addStorageImage(0, VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();
    genDesc.updateImage(0, imageA.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);

    auto genPipeline = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "gen_scene.comp.spv")
        .descriptorSetLayout(genDesc.vkDescriptorSetLayout())
        .pushConstants<float>(VK_SHADER_STAGE_COMPUTE_BIT)
        .build().value();

    // reflectDescriptors() reads the SPIR-V and creates:
    //   set 0, binding 0: storage image (readonly  -- inputImage)
    //   set 0, binding 1: storage image (writeonly -- outputImage)
    //   push constant range: 8 bytes (float time + float edgeStrength)
    // No manual descriptorSetLayout() or pushConstants() needed.
    auto filterPipeline = vksdl::ComputePipelineBuilder(device)
        .shader(shaderDir / "edge_filter.comp.spv")
        .reflectDescriptors()
        .build().value();

    // Print what reflection discovered.
    const auto& layouts = filterPipeline.reflectedSetLayouts();
    std::printf("Shader Reflection: edge_filter.comp\n");
    std::printf("  Reflected %zu descriptor set layout(s)\n", layouts.size());
    std::printf("  Bindings: set 0 binding 0 = storage image (input)\n");
    std::printf("            set 0 binding 1 = storage image (output)\n");
    std::printf("  Push constants: 8 bytes (float time, float edgeStrength)\n");
    std::printf("  -> No manual descriptorSetLayout() or pushConstants() calls.\n");
    std::printf("   The SPIR-V is the entire API contract.\n\n");

    // Allocate descriptor set from the reflected layout and populate it.
    auto pool = vksdl::DescriptorPool::create(device).value();
    VkDescriptorSet filterDS = pool.allocate(layouts[0]).value();
    writeFilterDescriptors(device.vkDevice(),
                           filterDS,
                           imageA.vkImageView(),
                           imageB.vkImageView());

    bool running = true;
    vksdl::Event event;
    auto startTime = std::chrono::steady_clock::now();
    int frameNum = 0;

    SDL_SetWindowTitle(window.sdlWindow(),
        "Shader Reflection | "
        "edge_filter.comp: 2 storage images + 8 B push constants (reflected)");

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);
            extent = swapchain.extent();

            imageA = makeStorageImage(allocator, extent).value();
            imageB = makeStorageImage(allocator, extent).value();

            genDesc.updateImage(0, imageA.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
            writeFilterDescriptors(device.vkDevice(),
                                   filterDS,
                                   imageA.vkImageView(),
                                   imageB.vkImageView());
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        auto  now  = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        VkCommandBuffer cmd = frame.cmd;
        vksdl::beginOneTimeCommands(cmd);

        vksdl::transitionToComputeWrite(cmd, imageA.vkImage());

        genPipeline.bind(cmd, genDesc);
        genPipeline.pushConstants(cmd, time);

        std::uint32_t groupsX = (extent.width  + 15) / 16;
        std::uint32_t groupsY = (extent.height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Compute-to-compute memory barrier: pass 1 write -> pass 2 read.
        VkMemoryBarrier2 memBarrier{};
        memBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        memBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                 | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo dep{};
        dep.sType               = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount  = 1;
        dep.pMemoryBarriers     = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        vksdl::transitionToComputeWrite(cmd, imageB.vkImage());

        // bind(cmd, VkDescriptorSet) -- raw handle overload, no wrapper needed
        filterPipeline.bind(cmd, filterDS);

        // pushConstants(cmd, T) -- uses stage flags recorded by reflection
        FilterPush filterPush{time, 3.0f};
        filterPipeline.pushConstants(cmd, filterPush);

        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vksdl::blitToSwapchain(cmd, imageB,
                               VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               img.image, swapchain.extent());

        vksdl::endCommands(cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        ++frameNum;
        if (frameNum % 60 == 0) {
            std::printf("[frame %d] gen_scene(manual) -> edge_filter(reflected), 2 compute passes\n",
                        frameNum);
        }
    }

    device.waitIdle();
    return 0;
}
