#include <vksdl/app.hpp>
#include <vksdl/device.hpp>
#include <vksdl/frames.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/surface.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("framesync test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_framesync")
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

    // Create FrameSync with 2 frames in flight
    auto frames = vksdl::FrameSync::create(device.value(), 2);
    assert(frames.ok());
    assert(frames.value().count() == 2);

    std::printf("  framesync created: %u frames in flight\n", frames.value().count());

    // Get frame 0 -- should not block (fences start signaled)
    auto f0 = frames.value().nextFrame();
    assert(f0.ok());
    assert(f0.value().cmd != VK_NULL_HANDLE);
    assert(f0.value().drawDone != VK_NULL_HANDLE);
    assert(f0.value().fence != VK_NULL_HANDLE);
    assert(f0.value().index == 0);
    std::printf("  frame 0: ok\n");

    vksdl::beginOneTimeCommands(f0.value().cmd);
    auto submit0 = vksdl::endSubmitOneShotBlocking(device.value().graphicsQueue(), f0.value().cmd,
                                                   f0.value().fence);
    assert(submit0.ok());
    std::printf("  frame 0 submit one-shot blocking: ok\n");

    // Get frame 1 -- different slot, should also not block
    auto f1 = frames.value().nextFrame();
    assert(f1.ok());
    assert(f1.value().index == 1);
    assert(f1.value().cmd != f0.value().cmd); // different command buffer
    assert(f1.value().drawDone != f0.value().drawDone);
    assert(f1.value().fence != f0.value().fence);
    std::printf("  frame 1: ok (different handles from frame 0)\n");

    vksdl::beginOneTimeCommands(f1.value().cmd);
    auto submit1 = vksdl::endSubmitOneShotBlocking(device.value().graphicsQueue(), f1.value().cmd,
                                                   f1.value().fence);
    assert(submit1.ok());
    std::printf("  frame 1 submit one-shot blocking: ok\n");

    auto f2 = frames.value().nextFrame();
    assert(f2.ok());
    assert(f2.value().index == 0);
    std::printf("  frame 2 after blocking submits: ok\n");

    // Clean up
    device.value().waitIdle();

    std::printf("framesync test passed\n");
    return 0;
}
