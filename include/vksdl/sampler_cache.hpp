#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_map>

namespace vksdl {

class Device;

// Deduplicating sampler cache. Hash the VkSamplerCreateInfo, return existing
// or create new. Owns all created VkSamplers -- destroyed in destructor.
//
// Typical scenes have 3-5 unique sampler configs but hundreds of materials.
// This eliminates the duplication every app writes manually.
//
// Thread safety: none (single-threaded init-time usage expected).
class SamplerCache {
  public:
    [[nodiscard]] static Result<SamplerCache> create(const Device& device);

    ~SamplerCache();
    SamplerCache(SamplerCache&&) noexcept;
    SamplerCache& operator=(SamplerCache&&) noexcept;
    SamplerCache(const SamplerCache&) = delete;
    SamplerCache& operator=(const SamplerCache&) = delete;

    // sType and pNext are ignored (always overwritten internally).
    [[nodiscard]] Result<VkSampler> get(const VkSamplerCreateInfo& ci);

    [[nodiscard]] std::uint32_t size() const {
        return static_cast<std::uint32_t>(cache_.size());
    }

  private:
    SamplerCache() = default;
    void destroy();

    // Hash key: the subset of VkSamplerCreateInfo fields that affect identity.
    struct SamplerKey {
        VkFilter magFilter;
        VkFilter minFilter;
        VkSamplerMipmapMode mipmapMode;
        VkSamplerAddressMode addressModeU;
        VkSamplerAddressMode addressModeV;
        VkSamplerAddressMode addressModeW;
        float mipLodBias;
        VkBool32 anisotropyEnable;
        float maxAnisotropy;
        VkBool32 compareEnable;
        VkCompareOp compareOp;
        float minLod;
        float maxLod;
        VkBorderColor borderColor;
        VkBool32 unnormalizedCoordinates;

        bool operator==(const SamplerKey&) const = default;
    };

    struct KeyHash {
        std::size_t operator()(const SamplerKey& k) const;
    };

    static SamplerKey toKey(const VkSamplerCreateInfo& ci);

    VkDevice device_ = VK_NULL_HANDLE;
    std::unordered_map<SamplerKey, VkSampler, KeyHash> cache_;
};

} // namespace vksdl
