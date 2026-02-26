#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vksdl {

enum class Validation {
    Off,
    On,
};

#ifdef NDEBUG
inline constexpr Validation DefaultValidation = Validation::Off;
#else
inline constexpr Validation DefaultValidation = Validation::On;
#endif

// Thread safety: immutable after construction.
class Instance {
public:
    ~Instance();
    Instance(Instance&&) noexcept;
    Instance& operator=(Instance&&) noexcept;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    [[nodiscard]] VkInstance native()     const { return instance_; }
    [[nodiscard]] VkInstance vkInstance() const { return native(); }
    [[nodiscard]] bool validationEnabled() const { return messenger_ != VK_NULL_HANDLE; }

private:
    friend class InstanceBuilder;
    Instance() = default;

    VkInstance               instance_  = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
};

class InstanceBuilder {
public:
    InstanceBuilder& appName(std::string_view name);
    InstanceBuilder& requireVulkan(std::uint32_t major, std::uint32_t minor, std::uint32_t patch = 0);
    InstanceBuilder& validation(Validation v);
    InstanceBuilder& addExtension(const char* name);
    InstanceBuilder& addLayer(const char* name);

    // Add WSI extensions for surface creation. No window needed (SDL3 knows).
    InstanceBuilder& enableWindowSupport();

    [[nodiscard]] Result<Instance> build();

private:
    std::string              appName_       = "vksdl_app";
    std::uint32_t            apiVersion_    = VK_API_VERSION_1_3;
    Validation               validation_    = DefaultValidation;
    std::vector<const char*> extensions_;
    std::vector<const char*> layers_;
    bool                     windowSupport_ = false;
};

} // namespace vksdl
