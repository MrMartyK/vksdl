#include <vksdl/app.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto appResult = vksdl::App::create();
    assert(appResult.ok());
    auto app = std::move(appResult.value());

    auto winResult = app.createWindow("instance builder test", 640, 480);
    assert(winResult.ok());
    auto window = std::move(winResult.value());

    // Build instance with window support + defaults
    {
        auto result = vksdl::InstanceBuilder{}
                          .appName("test_app")
                          .requireVulkan(1, 3)
                          .enableWindowSupport()
                          .build();

        assert(result.ok() && "build failed");
        auto instance = std::move(result.value());

        assert(instance.vkInstance() != VK_NULL_HANDLE);

#ifndef NDEBUG
        assert(instance.validationEnabled());
        std::printf("  validation: on\n");
#endif

        std::printf("  instance with window support: ok\n");
    }

    // Headless instance (no window support)
    {
        auto result = vksdl::InstanceBuilder{}
                          .appName("test_headless")
                          .requireVulkan(1, 3)
                          .validation(vksdl::Validation::Off)
                          .build();

        assert(result.ok());
        assert(result.value().vkInstance() != VK_NULL_HANDLE);
        std::printf("  headless instance: ok\n");
    }

    // Validation explicitly off
    {
        auto result = vksdl::InstanceBuilder{}
                          .appName("test_no_validation")
                          .requireVulkan(1, 3)
                          .validation(vksdl::Validation::Off)
                          .enableWindowSupport()
                          .build();

        assert(result.ok());
        assert(!result.value().validationEnabled());
        std::printf("  instance without validation: ok\n");
    }

    // Extra user extension
    {
        auto result = vksdl::InstanceBuilder{}
                          .appName("test_extra_ext")
                          .requireVulkan(1, 3)
                          .validation(vksdl::Validation::Off)
                          .addExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)
                          .enableWindowSupport()
                          .build();

        assert(result.ok());
        std::printf("  instance with extra extension: ok\n");
    }

    // Bogus extension rejected with readable error
    {
        auto result = vksdl::InstanceBuilder{}
                          .appName("test_bad_ext")
                          .validation(vksdl::Validation::Off)
                          .addExtension("VK_KHR_does_not_exist")
                          .enableWindowSupport()
                          .build();

        assert(!result.ok());
        auto msg = result.error().format();
        assert(msg.find("VK_KHR_does_not_exist") != std::string::npos);
        assert(msg.find("not available") != std::string::npos);
        std::printf("  bogus extension rejected: ok\n");
    }

    std::printf("instance builder test passed\n");
    return 0;
}
