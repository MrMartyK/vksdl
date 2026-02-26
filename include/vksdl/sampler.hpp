#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

namespace vksdl {

class Device;

// Thread safety: immutable after construction.
class Sampler {
public:
    ~Sampler();
    Sampler(Sampler&&) noexcept;
    Sampler& operator=(Sampler&&) noexcept;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    [[nodiscard]] VkSampler native()    const { return sampler_; }
    [[nodiscard]] VkSampler vkSampler() const { return native(); }

private:
    friend class SamplerBuilder;
    Sampler() = default;

    VkDevice  device_  = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

class SamplerBuilder {
public:
    explicit SamplerBuilder(const Device& device);

    SamplerBuilder& linear();        // mag=LINEAR, min=LINEAR, mipmap=LINEAR
    SamplerBuilder& nearest();       // mag=NEAREST, min=NEAREST, mipmap=NEAREST
    SamplerBuilder& repeat();        // U/V/W = REPEAT
    SamplerBuilder& clampToEdge();   // U/V/W = CLAMP_TO_EDGE
    SamplerBuilder& anisotropy(float maxAnisotropy);

    SamplerBuilder& magFilter(VkFilter filter);
    SamplerBuilder& minFilter(VkFilter filter);
    SamplerBuilder& addressModeU(VkSamplerAddressMode mode);
    SamplerBuilder& addressModeV(VkSamplerAddressMode mode);
    SamplerBuilder& addressModeW(VkSamplerAddressMode mode);
    SamplerBuilder& mipmapMode(VkSamplerMipmapMode mode);

    [[nodiscard]] Result<Sampler> build();

private:
    VkDevice             device_        = VK_NULL_HANDLE;
    VkFilter             magFilter_     = VK_FILTER_LINEAR;
    VkFilter             minFilter_     = VK_FILTER_LINEAR;
    VkSamplerAddressMode addressModeU_  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeV_  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeW_  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerMipmapMode  mipmapMode_    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    bool                 anisotropy_    = false;
    float                maxAnisotropy_ = 1.0f;
};

} // namespace vksdl
