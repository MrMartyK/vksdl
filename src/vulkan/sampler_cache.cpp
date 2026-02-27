#include <vksdl/device.hpp>
#include <vksdl/sampler_cache.hpp>

#include <cstring>
#include <functional>

namespace vksdl {

static std::size_t hashCombine(std::size_t seed, std::size_t v) {
    return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

std::size_t SamplerCache::KeyHash::operator()(const SamplerKey& k) const {
    std::size_t h = 0;
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.magFilter)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.minFilter)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.mipmapMode)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.addressModeU)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.addressModeV)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.addressModeW)));

    // Float hashing: memcpy to uint32 to avoid UB.
    auto hashFloat = [&](float f) {
        std::uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return std::hash<std::uint32_t>{}(bits);
    };

    h = hashCombine(h, hashFloat(k.mipLodBias));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.anisotropyEnable)));
    h = hashCombine(h, hashFloat(k.maxAnisotropy));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.compareEnable)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.compareOp)));
    h = hashCombine(h, hashFloat(k.minLod));
    h = hashCombine(h, hashFloat(k.maxLod));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.borderColor)));
    h = hashCombine(h, std::hash<int>{}(static_cast<int>(k.unnormalizedCoordinates)));
    return h;
}

SamplerCache::SamplerKey SamplerCache::toKey(const VkSamplerCreateInfo& ci) {
    return {
        ci.magFilter,     ci.minFilter,        ci.mipmapMode,
        ci.addressModeU,  ci.addressModeV,     ci.addressModeW,
        ci.mipLodBias,    ci.anisotropyEnable, ci.maxAnisotropy,
        ci.compareEnable, ci.compareOp,        ci.minLod,
        ci.maxLod,        ci.borderColor,      ci.unnormalizedCoordinates,
    };
}

void SamplerCache::destroy() {
    if (device_ == VK_NULL_HANDLE)
        return;
    for (auto& [key, sampler] : cache_)
        vkDestroySampler(device_, sampler, nullptr);
    cache_.clear();
    device_ = VK_NULL_HANDLE;
}

SamplerCache::~SamplerCache() {
    destroy();
}

SamplerCache::SamplerCache(SamplerCache&& o) noexcept
    : device_(o.device_), cache_(std::move(o.cache_)) {
    o.device_ = VK_NULL_HANDLE;
}

SamplerCache& SamplerCache::operator=(SamplerCache&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        cache_ = std::move(o.cache_);
        o.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<SamplerCache> SamplerCache::create(const Device& device) {
    SamplerCache sc;
    sc.device_ = device.vkDevice();
    return sc;
}

Result<VkSampler> SamplerCache::get(const VkSamplerCreateInfo& ci) {
    auto key = toKey(ci);
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    // Rebuild a clean create info from the key (ignore user's sType/pNext).
    VkSamplerCreateInfo cleanCI{};
    cleanCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    cleanCI.magFilter = ci.magFilter;
    cleanCI.minFilter = ci.minFilter;
    cleanCI.mipmapMode = ci.mipmapMode;
    cleanCI.addressModeU = ci.addressModeU;
    cleanCI.addressModeV = ci.addressModeV;
    cleanCI.addressModeW = ci.addressModeW;
    cleanCI.mipLodBias = ci.mipLodBias;
    cleanCI.anisotropyEnable = ci.anisotropyEnable;
    cleanCI.maxAnisotropy = ci.maxAnisotropy;
    cleanCI.compareEnable = ci.compareEnable;
    cleanCI.compareOp = ci.compareOp;
    cleanCI.minLod = ci.minLod;
    cleanCI.maxLod = ci.maxLod;
    cleanCI.borderColor = ci.borderColor;
    cleanCI.unnormalizedCoordinates = ci.unnormalizedCoordinates;

    VkSampler sampler = VK_NULL_HANDLE;
    VkResult vr = vkCreateSampler(device_, &cleanCI, nullptr, &sampler);
    if (vr != VK_SUCCESS) {
        return Error{"create cached sampler", static_cast<std::int32_t>(vr),
                     "vkCreateSampler failed in SamplerCache"};
    }

    cache_[key] = sampler;
    return sampler;
}

} // namespace vksdl
