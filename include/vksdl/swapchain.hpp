#pragma once

#include <vksdl/device.hpp>
#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/window.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

// Unified present timing record. Returned by Swapchain::queryPastPresentTiming().
// Maps from both VK_EXT_present_timing and VK_GOOGLE_display_timing.
struct PresentTiming {
    std::uint64_t presentId           = 0; // monotonic present counter
    std::uint64_t desiredPresentTime  = 0; // ns, 0 = ASAP
    std::uint64_t actualPresentTime   = 0; // ns, when frame was displayed
    std::uint64_t earliestPresentTime = 0; // ns, earliest the frame could have been displayed
    std::uint64_t presentMargin       = 0; // ns, slack between submission and display
};

// Image returned by nextImage().
struct SwapchainImage {
    std::uint32_t index      = 0;
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    VkSemaphore   imageReady = VK_NULL_HANDLE; // signaled when this image is acquired
};

enum class PresentMode {
    Fifo,           // vsync, always available
    Mailbox,        // triple-buffered vsync, preferred if available
    Immediate,      // no vsync
    MailboxOrFifo,  // try mailbox, fall back to fifo (default)
};

// RAII swapchain. Owns the VkSwapchainKHR, images, and image views.
// Supports recreate-on-resize.
//
// Thread safety: thread-confined (render loop thread).
class Swapchain {
public:
    ~Swapchain();
    Swapchain(Swapchain&&) noexcept;
    Swapchain& operator=(Swapchain&&) noexcept;
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    [[nodiscard]] VkSwapchainKHR             native()      const { return swapchain_; }
    [[nodiscard]] VkSwapchainKHR             vkSwapchain() const { return native(); }
    [[nodiscard]] VkFormat                    format()      const { return format_; }
    [[nodiscard]] VkExtent2D                  extent()      const { return extent_; }
    [[nodiscard]] const std::vector<VkImage>& images()      const { return images_; }
    [[nodiscard]] const std::vector<VkImageView>& imageViews() const { return views_; }
    [[nodiscard]] std::uint32_t              imageCount()   const { return static_cast<std::uint32_t>(images_.size()); }

    // Acquire the next image. Returns the image index + handles + the
    // semaphore that will be signaled when the image is ready.
    // Uses one semaphore per swapchain image internally (avoids reuse hazard).
    [[nodiscard]] Result<SwapchainImage> nextImage();

    // Present a rendered image.
    [[nodiscard]] VkResult present(VkQueue presentQueue,
                                   std::uint32_t imageIndex,
                                   VkSemaphore renderFinished);

    // Recreate after resize. Call after device.waitIdle().
    [[nodiscard]] Result<void> recreate(Size newSize);

    // Convenience: bundles device.waitIdle() + recreate(window.pixelSize()).
    [[nodiscard]] Result<void> recreate(const Device& device, const Window& window);

    // Present timing support. Mirrors device.hasPresentTiming().
    // When true, queryPastPresentTiming() returns actual display timestamps.
    [[nodiscard]] bool hasPresentTiming() const { return hasPresentTiming_; }

    // Query past present timing records from the driver.
    // Returns an empty vector when the extension is not available.
    [[nodiscard]] std::vector<PresentTiming> queryPastPresentTiming() const;

private:
    friend class SwapchainBuilder;
    Swapchain() = default;

    void destroyViews();
    void destroySemaphores();
    Result<void> createViews();
    Result<void> createSemaphores();

    VkDevice                device_    = VK_NULL_HANDLE;
    VkPhysicalDevice        gpu_       = VK_NULL_HANDLE;
    VkSurfaceKHR            surface_   = VK_NULL_HANDLE;
    VkSwapchainKHR          swapchain_ = VK_NULL_HANDLE;
    VkFormat                format_    = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR         colorSpace_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D              extent_    = {0, 0};
    VkPresentModeKHR        presentMode_ = VK_PRESENT_MODE_FIFO_KHR;
    std::uint32_t           imageCountRequested_ = 0;
    QueueFamilies           families_;
    std::vector<VkImage>      images_;
    std::vector<VkImageView>  views_;
    std::vector<VkSemaphore>  imageReadySems_; // one per swapchain image
    std::uint32_t             semIndex_ = 0;   // round-robin index

    // Present timing state
    bool hasPresentTiming_       = false;
    bool useGoogleDisplayTiming_ = false;
    PFN_vkGetPastPresentationTimingGOOGLE pfnGetPastTiming_ = nullptr;
    std::uint64_t presentCounter_ = 0; // monotonic counter for PresentTiming::presentId
};

class SwapchainBuilder {
public:
    SwapchainBuilder(const Device& device, const Surface& surface);

    SwapchainBuilder& size(Size windowPixelSize);
    SwapchainBuilder& forWindow(const Window& window);
    SwapchainBuilder& preferSrgb();
    SwapchainBuilder& colorSpace(VkColorSpaceKHR cs);
    SwapchainBuilder& presentMode(PresentMode mode);
    SwapchainBuilder& imageCount(std::uint32_t count);

    [[nodiscard]] Result<Swapchain> build();

private:
    VkDevice         device_      = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_         = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_     = VK_NULL_HANDLE;
    QueueFamilies    families_;
    Size             size_        = {0, 0};
    bool             preferSrgb_  = true;
    VkColorSpaceKHR  colorSpace_  = VK_COLOR_SPACE_MAX_ENUM_KHR; // MAX_ENUM = not set
    PresentMode      presentMode_ = PresentMode::MailboxOrFifo;
    std::uint32_t    imageCount_          = 0; // 0 = let builder choose (min+1)
    bool             hasPresentTiming_    = false;
    bool             useGoogleTiming_     = false;
};

} // namespace vksdl
