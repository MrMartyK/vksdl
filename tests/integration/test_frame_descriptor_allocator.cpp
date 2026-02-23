#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("frame descriptor allocator test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_frame_descriptor_allocator")
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

    // Create a simple layout for testing.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 1;
    layoutCI.pBindings    = &binding;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorSetLayout(device.value().vkDevice(),
                                               &layoutCI, nullptr, &layout);
    assert(vr == VK_SUCCESS);

    std::printf("frame descriptor allocator test\n");

    {
        auto fda = vksdl::FrameDescriptorAllocator::create(device.value(), 3);
        assert(fda.ok());
        assert(fda.value().frameCount() == 3);
        assert(fda.value().allocatedSetCount(0) == 0);
        assert(fda.value().allocatedSetCount(1) == 0);
        assert(fda.value().allocatedSetCount(2) == 0);
        std::printf("  1. create with 3 frames: ok\n");
    }

    {
        auto fda = vksdl::FrameDescriptorAllocator::create(device.value(), 2);
        assert(fda.ok());

        auto s0 = fda.value().allocate(0, layout);
        auto s1 = fda.value().allocate(1, layout);
        assert(s0.ok() && s1.ok());
        assert(s0.value() != VK_NULL_HANDLE);
        assert(s1.value() != VK_NULL_HANDLE);
        // Different frame indices -> different descriptor sets.
        assert(s0.value() != s1.value());
        assert(fda.value().allocatedSetCount(0) == 1);
        assert(fda.value().allocatedSetCount(1) == 1);
        std::printf("  2. allocate into different frames: ok\n");
    }

    {
        auto fda = vksdl::FrameDescriptorAllocator::create(device.value(), 2);
        assert(fda.ok());

        // Allocate into both frames.
        for (int i = 0; i < 5; ++i) {
            auto s = fda.value().allocate(0, layout);
            assert(s.ok());
        }
        for (int i = 0; i < 3; ++i) {
            auto s = fda.value().allocate(1, layout);
            assert(s.ok());
        }
        assert(fda.value().allocatedSetCount(0) == 5);
        assert(fda.value().allocatedSetCount(1) == 3);

        // Reset frame 0 only.
        fda.value().resetFrame(0);
        assert(fda.value().allocatedSetCount(0) == 0);
        assert(fda.value().allocatedSetCount(1) == 3);

        std::printf("  3. reset one frame, other unaffected: ok\n");
    }

    {
        constexpr std::uint32_t kFrames = 3;
        auto fda = vksdl::FrameDescriptorAllocator::create(device.value(), kFrames);
        assert(fda.ok());

        // Simulate 10 frame iterations with round-robin.
        for (std::uint32_t frame = 0; frame < 10; ++frame) {
            std::uint32_t idx = frame % kFrames;

            // Reset this frame's allocator (simulates post-fence-wait).
            fda.value().resetFrame(idx);
            assert(fda.value().allocatedSetCount(idx) == 0);

            // Allocate 4 sets (typical: per-pass descriptors).
            for (int i = 0; i < 4; ++i) {
                auto s = fda.value().allocate(idx, layout);
                assert(s.ok());
                assert(s.value() != VK_NULL_HANDLE);
            }
            assert(fda.value().allocatedSetCount(idx) == 4);
        }
        std::printf("  4. simulated frame loop (10 frames): ok\n");
    }

    {
        auto fda = vksdl::FrameDescriptorAllocator::create(device.value(), 2);
        assert(fda.ok());

        auto s = fda.value().allocate(0, layout);
        assert(s.ok());

        // Move-construct.
        vksdl::FrameDescriptorAllocator moved = std::move(fda.value());
        assert(moved.frameCount() == 2);
        assert(moved.allocatedSetCount(0) == 1);

        // Move-assign.
        auto fda2 = vksdl::FrameDescriptorAllocator::create(device.value(), 3);
        assert(fda2.ok());
        fda2.value() = std::move(moved);
        assert(fda2.value().frameCount() == 2);
        assert(fda2.value().allocatedSetCount(0) == 1);

        std::printf("  5. move semantics: ok\n");
    }

    {
        // Small pool (32 sets) to force chaining.
        auto fda = vksdl::FrameDescriptorAllocator::create(device.value(), 2, 32);
        assert(fda.ok());

        // Allocate 100 sets into frame 0 -- forces pool chain growth.
        for (int i = 0; i < 100; ++i) {
            auto s = fda.value().allocate(0, layout);
            assert(s.ok());
        }
        assert(fda.value().allocatedSetCount(0) == 100);

        // Frame 1 is untouched.
        assert(fda.value().allocatedSetCount(1) == 0);

        // Reset and reallocate -- pools reused.
        fda.value().resetFrame(0);
        for (int i = 0; i < 100; ++i) {
            auto s = fda.value().allocate(0, layout);
            assert(s.ok());
        }
        assert(fda.value().allocatedSetCount(0) == 100);
        std::printf("  6. bulk allocation with pool chaining: ok\n");
    }

    vkDestroyDescriptorSetLayout(device.value().vkDevice(), layout, nullptr);

    device.value().waitIdle();
    std::printf("frame descriptor allocator test passed\n");
    return 0;
}
