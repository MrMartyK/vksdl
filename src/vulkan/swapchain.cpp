#include <vksdl/swapchain.hpp>
#include <vksdl/surface.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace vksdl {

Swapchain::~Swapchain() {
    destroySemaphores();
    destroyViews();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }
}

Swapchain::Swapchain(Swapchain&& o) noexcept
    : device_(o.device_), gpu_(o.gpu_), surface_(o.surface_),
      swapchain_(o.swapchain_), format_(o.format_),
      colorSpace_(o.colorSpace_), extent_(o.extent_),
      presentMode_(o.presentMode_),
      imageCountRequested_(o.imageCountRequested_),
      families_(o.families_),
      images_(std::move(o.images_)), views_(std::move(o.views_)),
      imageReadySems_(std::move(o.imageReadySems_)), semIndex_(o.semIndex_),
      hasPresentTiming_(o.hasPresentTiming_),
      useGoogleDisplayTiming_(o.useGoogleDisplayTiming_),
      pfnGetPastTiming_(o.pfnGetPastTiming_),
      presentCounter_(o.presentCounter_),
      googlePresentId_(o.googlePresentId_) {
    o.swapchain_       = VK_NULL_HANDLE;
    o.device_          = VK_NULL_HANDLE;
    o.pfnGetPastTiming_ = nullptr;
}

Swapchain& Swapchain::operator=(Swapchain&& o) noexcept {
    if (this != &o) {
        destroySemaphores();
        destroyViews();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        }
        device_    = o.device_;
        gpu_       = o.gpu_;
        surface_   = o.surface_;
        swapchain_ = o.swapchain_;
        format_    = o.format_;
        colorSpace_ = o.colorSpace_;
        extent_    = o.extent_;
        presentMode_ = o.presentMode_;
        imageCountRequested_ = o.imageCountRequested_;
        families_  = o.families_;
        images_    = std::move(o.images_);
        views_     = std::move(o.views_);
        imageReadySems_         = std::move(o.imageReadySems_);
        semIndex_               = o.semIndex_;
        hasPresentTiming_       = o.hasPresentTiming_;
        useGoogleDisplayTiming_ = o.useGoogleDisplayTiming_;
        pfnGetPastTiming_       = o.pfnGetPastTiming_;
        presentCounter_         = o.presentCounter_;
        googlePresentId_        = o.googlePresentId_;
        o.swapchain_        = VK_NULL_HANDLE;
        o.device_           = VK_NULL_HANDLE;
        o.pfnGetPastTiming_ = nullptr;
    }
    return *this;
}

void Swapchain::destroyViews() {
    for (auto v : views_) {
        if (v != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, v, nullptr);
        }
    }
    views_.clear();
}

Result<void> Swapchain::createViews() {
    views_.resize(images_.size());
    for (std::size_t i = 0; i < images_.size(); ++i) {
        VkImageViewCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image    = images_[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format   = format_;
        ci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.baseMipLevel   = 0;
        ci.subresourceRange.levelCount     = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount     = 1;

        VkResult vr = vkCreateImageView(device_, &ci, nullptr, &views_[i]);
        if (vr != VK_SUCCESS) {
            destroyViews();
            return Error{"create swapchain image view",
                         static_cast<std::int32_t>(vr),
                         "vkCreateImageView failed for swapchain image " +
                         std::to_string(i)};
        }
    }
    return {};
}

void Swapchain::destroySemaphores() {
    for (auto s : imageReadySems_) {
        if (s != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, s, nullptr);
        }
    }
    imageReadySems_.clear();
}

Result<void> Swapchain::createSemaphores() {
    destroySemaphores();
    imageReadySems_.resize(images_.size());

    VkSemaphoreCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (std::size_t i = 0; i < images_.size(); ++i) {
        VkResult vr = vkCreateSemaphore(device_, &ci, nullptr, &imageReadySems_[i]);
        if (vr != VK_SUCCESS) {
            destroySemaphores();
            return Error{"create swapchain semaphore",
                         static_cast<std::int32_t>(vr),
                         "vkCreateSemaphore failed for image " + std::to_string(i)};
        }
    }
    semIndex_ = 0;
    return {};
}

Result<SwapchainImage> Swapchain::nextImage() {
    // Pick the next semaphore in round-robin order.
    VkSemaphore sem = imageReadySems_[semIndex_];
    semIndex_ = (semIndex_ + 1) % static_cast<std::uint32_t>(imageReadySems_.size());

    std::uint32_t index = 0;
    VkResult vr = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                         sem, VK_NULL_HANDLE, &index);

    if (vr == VK_ERROR_OUT_OF_DATE_KHR) {
        return Error{"acquire image", static_cast<std::int32_t>(vr),
                     "swapchain out of date -- call recreate()"};
    }
    if (vr != VK_SUCCESS && vr != VK_SUBOPTIMAL_KHR) {
        return Error{"acquire image", static_cast<std::int32_t>(vr),
                     "vkAcquireNextImageKHR failed"};
    }

    return SwapchainImage{index, images_[index], views_[index], sem};
}

VkResult Swapchain::present(VkQueue presentQueue,
                            std::uint32_t imageIndex,
                            VkSemaphore renderFinished) {
    ++presentCounter_;

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderFinished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &imageIndex;

    // Chain VkPresentTimesInfoGOOGLE when using VK_GOOGLE_display_timing so
    // the driver can correlate presentId with the recorded timing data.
    VkPresentTimeGOOGLE presentTime{};
    VkPresentTimesInfoGOOGLE timesInfo{};
    if (useGoogleDisplayTiming_) {
        presentTime.presentID       = ++googlePresentId_;
        presentTime.desiredPresentTime = 0; // ASAP
        timesInfo.sType             = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE;
        timesInfo.pNext             = nullptr;
        timesInfo.swapchainCount    = 1;
        timesInfo.pTimes            = &presentTime;
        pi.pNext = &timesInfo;
    }

    return vkQueuePresentKHR(presentQueue, &pi);
}

std::vector<PresentTiming> Swapchain::queryPastPresentTiming() const {
    if (!hasPresentTiming_ || swapchain_ == VK_NULL_HANDLE) {
        return {};
    }

    if (useGoogleDisplayTiming_ && pfnGetPastTiming_) {
        std::uint32_t count = 0;
        VkResult vr = pfnGetPastTiming_(device_, swapchain_, &count, nullptr);
        if (vr != VK_SUCCESS || count == 0) return {};

        std::vector<VkPastPresentationTimingGOOGLE> raw(count);
        vr = pfnGetPastTiming_(device_, swapchain_, &count, raw.data());
        if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) return {};

        std::vector<PresentTiming> result;
        result.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            PresentTiming pt;
            pt.presentId           = raw[i].presentID;
            pt.desiredPresentTime  = raw[i].desiredPresentTime;
            pt.actualPresentTime   = raw[i].actualPresentTime;
            pt.earliestPresentTime = raw[i].earliestPresentTime;
            pt.presentMargin       = raw[i].presentMargin;
            result.push_back(pt);
        }
        return result;
    }

    // VK_EXT_present_timing: function pointers not yet defined in SDK headers.
    // Return empty until SDK support is available.
    return {};
}

Result<void> Swapchain::recreate(Size newSize) {
    if (newSize.width == 0 || newSize.height == 0) {
        return {}; // minimized/no drawable area
    }

    destroySemaphores();
    destroyViews();

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surface_, &caps);

    extent_.width  = std::clamp(newSize.width,
                                caps.minImageExtent.width,
                                caps.maxImageExtent.width);
    extent_.height = std::clamp(newSize.height,
                                caps.minImageExtent.height,
                                caps.maxImageExtent.height);

    if (extent_.width == 0 || extent_.height == 0) {
        return {}; // minimized, skip
    }

    VkSwapchainKHR oldSwapchain = swapchain_;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface_;
    ci.minImageCount    = imageCountRequested_;
    ci.imageFormat      = format_;
    ci.imageColorSpace  = colorSpace_;
    ci.imageExtent      = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = presentMode_;
    ci.clipped          = VK_TRUE;
    ci.oldSwapchain     = oldSwapchain;

    std::uint32_t familyIndices[] = {families_.graphics, families_.present};
    if (families_.shared()) {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    } else {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = familyIndices;
    }

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    VkResult vr = vkCreateSwapchainKHR(device_, &ci, nullptr, &newSwapchain);

    if (vr != VK_SUCCESS) {
        // Restore old swapchain so the object remains usable.
        swapchain_ = oldSwapchain;
        return Error{"recreate swapchain", static_cast<std::int32_t>(vr),
                     "vkCreateSwapchainKHR failed during recreate"};
    }

    swapchain_ = newSwapchain;
    if (oldSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, oldSwapchain, nullptr);
    }

    std::uint32_t count = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
    images_.resize(count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &count, images_.data());

    auto viewRes = createViews();
    if (!viewRes.ok()) return viewRes.error();

    return createSemaphores();
}

Result<void> Swapchain::recreate(const Device& device, const Window& window) {
    device.waitIdle();
    return recreate(window.pixelSize());
}

SwapchainBuilder::SwapchainBuilder(const Device& device, const Surface& surface)
    : device_(device.vkDevice()),
      gpu_(device.vkPhysicalDevice()),
      surface_(surface.vkSurface()),
      families_(device.queueFamilies()),
      hasPresentTiming_(device.hasPresentTiming()),
      // Prefer VK_EXT_present_timing (newer) -- fall back to GOOGLE when only that is available.
      useGoogleTiming_(device.hasPresentTiming() && !device.hasExtPresentTiming()) {}

SwapchainBuilder& SwapchainBuilder::size(Size windowPixelSize) {
    size_ = windowPixelSize;
    return *this;
}

SwapchainBuilder& SwapchainBuilder::forWindow(const Window& window) {
    return size(window.pixelSize());
}

SwapchainBuilder& SwapchainBuilder::preferSrgb() {
    preferSrgb_ = true;
    return *this;
}

SwapchainBuilder& SwapchainBuilder::colorSpace(VkColorSpaceKHR cs) {
    colorSpace_ = cs;
    return *this;
}

SwapchainBuilder& SwapchainBuilder::presentMode(PresentMode mode) {
    presentMode_ = mode;
    return *this;
}

SwapchainBuilder& SwapchainBuilder::imageCount(std::uint32_t count) {
    imageCount_ = count;
    return *this;
}

Result<Swapchain> SwapchainBuilder::build() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surface_, &caps);

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &formatCount, nullptr);
    if (formatCount == 0) {
        return Error{"create swapchain", 0, "No surface formats available"};
    }
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &formatCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    if (preferSrgb_) {
        for (auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                break;
            }
        }
    }
    // colorSpace() override: find a format with the requested color space,
    // preferring the already-chosen format if available in that color space.
    if (colorSpace_ != VK_COLOR_SPACE_MAX_ENUM_KHR) {
        for (auto& f : formats) {
            if (f.colorSpace == colorSpace_ && f.format == chosen.format) {
                chosen = f;
                break;
            }
        }
        // Fall back to any format with the requested color space.
        if (chosen.colorSpace != colorSpace_) {
            for (auto& f : formats) {
                if (f.colorSpace == colorSpace_) {
                    chosen = f;
                    break;
                }
            }
        }
    }

    std::uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &modeCount, modes.data());

    VkPresentModeKHR vkMode = VK_PRESENT_MODE_FIFO_KHR; // always available
    auto hasMode = [&](VkPresentModeKHR m) {
        return std::find(modes.begin(), modes.end(), m) != modes.end();
    };

    switch (presentMode_) {
    case PresentMode::Fifo:
        vkMode = VK_PRESENT_MODE_FIFO_KHR;
        break;
    case PresentMode::Mailbox:
        if (hasMode(VK_PRESENT_MODE_MAILBOX_KHR))
            vkMode = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
    case PresentMode::Immediate:
        if (hasMode(VK_PRESENT_MODE_IMMEDIATE_KHR))
            vkMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        break;
    case PresentMode::MailboxOrFifo:
        if (hasMode(VK_PRESENT_MODE_MAILBOX_KHR))
            vkMode = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
    }

    VkExtent2D extent;
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        extent.width  = std::clamp(size_.width,
                                   caps.minImageExtent.width,
                                   caps.maxImageExtent.width);
        extent.height = std::clamp(size_.height,
                                   caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }

    std::uint32_t imgCount = imageCount_;
    if (imgCount == 0) {
        imgCount = caps.minImageCount + 1;
    }
    if (caps.maxImageCount > 0 && imgCount > caps.maxImageCount) {
        imgCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface_;
    ci.minImageCount    = imgCount;
    ci.imageFormat      = chosen.format;
    ci.imageColorSpace  = chosen.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = vkMode;
    ci.clipped          = VK_TRUE;
    ci.oldSwapchain     = VK_NULL_HANDLE;

    std::uint32_t familyIndices[] = {families_.graphics, families_.present};
    if (families_.shared()) {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    } else {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = familyIndices;
    }

    Swapchain sc;
    sc.device_    = device_;
    sc.gpu_       = gpu_;
    sc.surface_   = surface_;
    sc.format_    = chosen.format;
    sc.colorSpace_ = chosen.colorSpace;
    sc.extent_    = extent;
    sc.presentMode_ = vkMode;
    sc.imageCountRequested_ = imgCount;
    sc.families_  = families_;
    sc.hasPresentTiming_       = hasPresentTiming_;
    sc.useGoogleDisplayTiming_ = useGoogleTiming_;
    if (useGoogleTiming_) {
        sc.pfnGetPastTiming_ = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
            vkGetDeviceProcAddr(device_, "vkGetPastPresentationTimingGOOGLE"));
    }

    VkResult vr = vkCreateSwapchainKHR(device_, &ci, nullptr, &sc.swapchain_);
    if (vr != VK_SUCCESS) {
        return Error{"create swapchain", static_cast<std::int32_t>(vr),
                     "vkCreateSwapchainKHR failed"};
    }

    std::uint32_t count = 0;
    vkGetSwapchainImagesKHR(device_, sc.swapchain_, &count, nullptr);
    sc.images_.resize(count);
    vkGetSwapchainImagesKHR(device_, sc.swapchain_, &count, sc.images_.data());

    auto viewResult = sc.createViews();
    if (!viewResult.ok()) return viewResult.error();

    auto semResult = sc.createSemaphores();
    if (!semResult.ok()) return semResult.error();

    return sc;
}

} // namespace vksdl
