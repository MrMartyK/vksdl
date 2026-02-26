#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <filesystem>

namespace vksdl {

class Device;

// RAII wrapper for VkPipelineCache with disk persistence.
// Move-only. Destroys the cache on destruction.
class PipelineCache {
public:
    ~PipelineCache();
    PipelineCache(PipelineCache&&) noexcept;
    PipelineCache& operator=(PipelineCache&&) noexcept;
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    [[nodiscard]] static Result<PipelineCache> create(const Device& device);

    // Falls back to empty cache if the file does not exist or is unreadable.
    [[nodiscard]] static Result<PipelineCache> load(const Device& device,
                                                     const std::filesystem::path& path);

    [[nodiscard]] Result<void> save(const std::filesystem::path& path) const;
    [[nodiscard]] Result<void> merge(const PipelineCache& src);
    [[nodiscard]] Result<void> merge(VkPipelineCache src);

    [[nodiscard]] std::size_t dataSize() const;

    [[nodiscard]] VkPipelineCache vkPipelineCache() const { return cache_; }

private:
    PipelineCache() = default;

    VkDevice         device_ = VK_NULL_HANDLE;
    VkPipelineCache  cache_  = VK_NULL_HANDLE;
};

} // namespace vksdl
