#include <vksdl/device.hpp>
#include <vksdl/pipeline_cache.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>

namespace vksdl {

PipelineCache::~PipelineCache() {
    if (cache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(device_, cache_, nullptr);
    }
}

PipelineCache::PipelineCache(PipelineCache&& o) noexcept : device_(o.device_), cache_(o.cache_) {
    o.device_ = VK_NULL_HANDLE;
    o.cache_ = VK_NULL_HANDLE;
}

PipelineCache& PipelineCache::operator=(PipelineCache&& o) noexcept {
    if (this != &o) {
        if (cache_ != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device_, cache_, nullptr);
        }
        device_ = o.device_;
        cache_ = o.cache_;
        o.device_ = VK_NULL_HANDLE;
        o.cache_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<PipelineCache> PipelineCache::create(const Device& device) {
    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

    PipelineCache pc;
    pc.device_ = device.vkDevice();

    VkResult vr = vkCreatePipelineCache(pc.device_, &ci, nullptr, &pc.cache_);
    if (vr != VK_SUCCESS) {
        return Error{"create pipeline cache", static_cast<std::int32_t>(vr),
                     "vkCreatePipelineCache failed"};
    }

    return pc;
}

// Falls back to empty cache when the file is missing or incompatible with the current driver.
Result<PipelineCache> PipelineCache::load(const Device& device, const std::filesystem::path& path) {
    std::vector<std::uint8_t> blob;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        auto size = static_cast<std::size_t>(file.tellg());
        if (size > 0) {
            blob.resize(size);
            file.seekg(0);
            file.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(size));
        }
    }
    // If the file doesn't exist or is empty, we create an empty cache.
    // The driver validates the blob and ignores it if it's incompatible.

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = blob.size();
    ci.pInitialData = blob.empty() ? nullptr : blob.data();

    PipelineCache pc;
    pc.device_ = device.vkDevice();

    VkResult vr = vkCreatePipelineCache(pc.device_, &ci, nullptr, &pc.cache_);
    if (vr != VK_SUCCESS) {
        return Error{"load pipeline cache", static_cast<std::int32_t>(vr),
                     "vkCreatePipelineCache failed with cached data from: " + path.string()};
    }

    return pc;
}

Result<void> PipelineCache::save(const std::filesystem::path& path) const {
    std::size_t dataSize = 0;
    VkResult vr = vkGetPipelineCacheData(device_, cache_, &dataSize, nullptr);
    if (vr != VK_SUCCESS) {
        return Error{"save pipeline cache", static_cast<std::int32_t>(vr),
                     "vkGetPipelineCacheData (query size) failed"};
    }

    std::vector<std::uint8_t> blob(dataSize);
    vr = vkGetPipelineCacheData(device_, cache_, &dataSize, blob.data());
    if (vr != VK_SUCCESS) {
        return Error{"save pipeline cache", static_cast<std::int32_t>(vr),
                     "vkGetPipelineCacheData (retrieve data) failed"};
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return Error{"save pipeline cache", 0, "could not open file for writing: " + path.string()};
    }

    file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(dataSize));
    if (!file.good()) {
        return Error{"save pipeline cache", 0, "write failed: " + path.string()};
    }

    return {};
}

Result<void> PipelineCache::merge(const PipelineCache& src) {
    return merge(src.vkPipelineCache());
}

Result<void> PipelineCache::merge(VkPipelineCache src) {
    if (src == VK_NULL_HANDLE) {
        return Error{"merge pipeline cache", 0, "source cache is null"};
    }
    if (src == cache_) {
        return {};
    }

    VkResult vr = vkMergePipelineCaches(device_, cache_, 1, &src);
    if (vr != VK_SUCCESS) {
        return Error{"merge pipeline cache", static_cast<std::int32_t>(vr),
                     "vkMergePipelineCaches failed"};
    }

    return {};
}

std::size_t PipelineCache::dataSize() const {
    if (cache_ == VK_NULL_HANDLE)
        return 0;

    std::size_t size = 0;
    vkGetPipelineCacheData(device_, cache_, &size, nullptr);
    return size;
}

} // namespace vksdl
