#pragma once
#include <vulkan/vulkan.h>

namespace vksdl {
class Device;
} // namespace vksdl

namespace vksdl::detail {

// Check if vr is VK_ERROR_DEVICE_LOST and trigger the device's error recovery.
// Returns true if device was lost (caller should propagate the error).
inline bool checkDeviceLost(const Device& device, VkResult vr) {
    if (vr == VK_ERROR_DEVICE_LOST) {
        device.reportDeviceLost();
        return true;
    }
    return false;
}

} // namespace vksdl::detail
