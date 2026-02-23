#include <vksdl/sampler.hpp>
#include <vksdl/device.hpp>

namespace vksdl {

Sampler::~Sampler() {
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, sampler_, nullptr);
    }
}

Sampler::Sampler(Sampler&& o) noexcept
    : device_(o.device_), sampler_(o.sampler_) {
    o.device_  = VK_NULL_HANDLE;
    o.sampler_ = VK_NULL_HANDLE;
}

Sampler& Sampler::operator=(Sampler&& o) noexcept {
    if (this != &o) {
        if (sampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, sampler_, nullptr);
        }
        device_  = o.device_;
        sampler_ = o.sampler_;
        o.device_  = VK_NULL_HANDLE;
        o.sampler_ = VK_NULL_HANDLE;
    }
    return *this;
}

SamplerBuilder::SamplerBuilder(const Device& device)
    : device_(device.vkDevice()) {}

SamplerBuilder& SamplerBuilder::linear() {
    magFilter_  = VK_FILTER_LINEAR;
    minFilter_  = VK_FILTER_LINEAR;
    mipmapMode_ = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    return *this;
}

SamplerBuilder& SamplerBuilder::nearest() {
    magFilter_  = VK_FILTER_NEAREST;
    minFilter_  = VK_FILTER_NEAREST;
    mipmapMode_ = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    return *this;
}

SamplerBuilder& SamplerBuilder::repeat() {
    addressModeU_ = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    addressModeV_ = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    addressModeW_ = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    return *this;
}

SamplerBuilder& SamplerBuilder::clampToEdge() {
    addressModeU_ = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    addressModeV_ = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    addressModeW_ = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    return *this;
}

SamplerBuilder& SamplerBuilder::anisotropy(float maxAniso) {
    anisotropy_    = true;
    maxAnisotropy_ = maxAniso;
    return *this;
}

SamplerBuilder& SamplerBuilder::magFilter(VkFilter filter) {
    magFilter_ = filter;
    return *this;
}

SamplerBuilder& SamplerBuilder::minFilter(VkFilter filter) {
    minFilter_ = filter;
    return *this;
}

SamplerBuilder& SamplerBuilder::addressModeU(VkSamplerAddressMode mode) {
    addressModeU_ = mode;
    return *this;
}

SamplerBuilder& SamplerBuilder::addressModeV(VkSamplerAddressMode mode) {
    addressModeV_ = mode;
    return *this;
}

SamplerBuilder& SamplerBuilder::addressModeW(VkSamplerAddressMode mode) {
    addressModeW_ = mode;
    return *this;
}

SamplerBuilder& SamplerBuilder::mipmapMode(VkSamplerMipmapMode mode) {
    mipmapMode_ = mode;
    return *this;
}

Result<Sampler> SamplerBuilder::build() {
    VkSamplerCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter    = magFilter_;
    ci.minFilter    = minFilter_;
    ci.addressModeU = addressModeU_;
    ci.addressModeV = addressModeV_;
    ci.addressModeW = addressModeW_;
    ci.mipmapMode   = mipmapMode_;
    ci.maxLod       = VK_LOD_CLAMP_NONE;

    if (anisotropy_) {
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy    = maxAnisotropy_;
    }

    Sampler s;
    s.device_ = device_;

    VkResult vr = vkCreateSampler(device_, &ci, nullptr, &s.sampler_);
    if (vr != VK_SUCCESS) {
        return Error{"create sampler", static_cast<std::int32_t>(vr),
                     "vkCreateSampler failed"};
    }

    return s;
}

} // namespace vksdl
