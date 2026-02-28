#include <vksdl/vksdl.hpp>

#include <atomic>
#include <cassert>
#include <cstdio>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("error recovery test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_error_recovery")
                        .requireVulkan(1, 3)
                        .validation(vksdl::Validation::Off)
                        .enableWindowSupport()
                        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    auto deviceRes = vksdl::DeviceBuilder(instance.value(), surface.value())
                         .needSwapchain()
                         .needDynamicRendering()
                         .needSync2()
                         .preferDiscreteGpu()
                         .build();
    assert(deviceRes.ok());

    vksdl::Device& device = deviceRes.value();

    // Test 1: isDeviceLost() returns false on a healthy device.
    assert(!device.isDeviceLost() && "new device must not be in lost state");
    std::printf("  test 1 passed: isDeviceLost() == false initially\n");

    // Test 2: custom callback is stored and invoked by reportDeviceLost().
    // The callback sets a flag instead of aborting so the test can continue.
    bool callbackInvoked = false;
    const vksdl::Device* callbackDevicePtr = nullptr;

    device.onDeviceLost([&callbackInvoked, &callbackDevicePtr](const vksdl::Device& d) {
        callbackInvoked = true;
        callbackDevicePtr = &d;
        // Do not abort -- test needs to verify state after the call.
    });

    std::printf("  test 2 setup: custom callback registered\n");

    // Test 3: reportDeviceLost() sets deviceLost_ and invokes the callback.
    device.reportDeviceLost();

    assert(callbackInvoked && "callback must be invoked by reportDeviceLost()");
    std::printf("  test 3 passed: callback was invoked\n");

    // Test 4: isDeviceLost() returns true after reportDeviceLost().
    assert(device.isDeviceLost() && "isDeviceLost() must be true after reportDeviceLost()");
    std::printf("  test 4 passed: isDeviceLost() == true after reportDeviceLost()\n");

    // Test 5: callback received a reference to the same device.
    assert(callbackDevicePtr == &device && "callback must receive reference to the device");
    std::printf("  test 5 passed: callback received correct device reference\n");

    // Test 6: move semantics transfer deviceLost_ and callback correctly.
    // Create a second device to test move with fresh state.
    auto deviceRes2 = vksdl::DeviceBuilder(instance.value(), surface.value())
                          .needSwapchain()
                          .needDynamicRendering()
                          .needSync2()
                          .preferDiscreteGpu()
                          .build();
    assert(deviceRes2.ok());

    vksdl::Device& device2 = deviceRes2.value();
    assert(!device2.isDeviceLost() && "second device must start clean");

    bool moved_callback_invoked = false;
    device2.onDeviceLost(
        [&moved_callback_invoked](const vksdl::Device&) { moved_callback_invoked = true; });

    vksdl::Device device2_moved = std::move(device2);
    assert(!device2_moved.isDeviceLost() && "moved device must carry over lost state (false)");

    device2_moved.reportDeviceLost();
    assert(moved_callback_invoked && "moved device must retain the callback");
    assert(device2_moved.isDeviceLost() && "moved device must reflect lost state");
    std::printf("  test 6 passed: move semantics preserve callback and state\n");

    std::printf("error recovery test passed\n");
    return 0;
}
