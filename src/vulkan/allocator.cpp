#include <vksdl/allocator.hpp>
#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

namespace vksdl {

Allocator::~Allocator() {
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }
}

Allocator::Allocator(Allocator&& o) noexcept
    : allocator_(o.allocator_), device_(o.device_) {
    o.allocator_ = nullptr;
    o.device_    = VK_NULL_HANDLE;
}

Allocator& Allocator::operator=(Allocator&& o) noexcept {
    if (this != &o) {
        if (allocator_ != nullptr) {
            vmaDestroyAllocator(allocator_);
        }
        allocator_   = o.allocator_;
        device_      = o.device_;
        o.allocator_ = nullptr;
        o.device_    = VK_NULL_HANDLE;
    }
    return *this;
}

Result<Allocator> Allocator::create(const Instance& instance, const Device& device) {
    VmaAllocatorCreateInfo ci{};
    ci.instance         = instance.vkInstance();
    ci.physicalDevice   = device.vkPhysicalDevice();
    ci.device           = device.vkDevice();
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    // Always enable BDA support. bufferDeviceAddress is a required Vulkan 1.2
    // feature (promoted from VK_KHR_buffer_device_address), so every Vulkan 1.3
    // driver supports it. Zero overhead when BDA is not actually used. Avoids
    // a footgun where RT code silently fails without this flag.
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    Allocator a;
    a.device_ = device.vkDevice();

    VkResult vr = vmaCreateAllocator(&ci, &a.allocator_);
    if (vr != VK_SUCCESS) {
        return Error{"create allocator", static_cast<std::int32_t>(vr),
                     "vmaCreateAllocator failed"};
    }

    return a;
}

} // namespace vksdl
