#include <vksdl/error.hpp>

#include <cassert>
#include <string>

int main() {
    // Vulkan error with message
    {
        vksdl::Error e{"create device", -3, "GPU only supports Vulkan 1.2"};
        std::string s = e.format();
        assert(s.find("create device") != std::string::npos);
        assert(s.find("VkResult -3") != std::string::npos);
        assert(s.find("GPU only supports Vulkan 1.2") != std::string::npos);
    }

    // Non-Vulkan error (vkResult == 0)
    {
        vksdl::Error e{"open window", 0, "display not available"};
        std::string s = e.format();
        assert(s.find("open window") != std::string::npos);
        assert(s.find("VkResult") == std::string::npos);
        assert(s.find("display not available") != std::string::npos);
    }

    // No message
    {
        vksdl::Error e{"create instance", -1, ""};
        std::string s = e.format();
        assert(s.find("create instance") != std::string::npos);
        assert(s.find("VkResult -1") != std::string::npos);
    }

    return 0;
}
