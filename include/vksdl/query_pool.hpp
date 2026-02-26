#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;

// Thread safety: thread-confined.
class QueryPool {
public:
    [[nodiscard]] static Result<QueryPool> create(const Device& device,
                                                  VkQueryType type,
                                                  std::uint32_t count);

    ~QueryPool();
    QueryPool(QueryPool&&) noexcept;
    QueryPool& operator=(QueryPool&&) noexcept;
    QueryPool(const QueryPool&) = delete;
    QueryPool& operator=(const QueryPool&) = delete;

    [[nodiscard]] VkQueryPool     native()      const { return pool_; }
    [[nodiscard]] VkQueryPool     vkQueryPool() const { return native(); }
    [[nodiscard]] VkQueryType     type()        const { return type_; }
    [[nodiscard]] std::uint32_t   count()       const { return count_; }

    // VK_QUERY_RESULT_64_BIT is always ORed into flags.
    // Add VK_QUERY_RESULT_WAIT_BIT to block until results are available.
    [[nodiscard]] Result<std::vector<std::uint64_t>> getResults(
        std::uint32_t first, std::uint32_t resultCount,
        VkQueryResultFlags flags = 0) const;

private:
    QueryPool() = default;

    VkDevice      device_ = VK_NULL_HANDLE;
    VkQueryPool   pool_   = VK_NULL_HANDLE;
    VkQueryType   type_   = VK_QUERY_TYPE_TIMESTAMP;
    std::uint32_t count_  = 0;
};

// Reset queries before use. Wraps vkCmdResetQueryPool.
void resetQueries(VkCommandBuffer cmd, const QueryPool& pool,
                  std::uint32_t first, std::uint32_t count);

// Write a GPU timestamp. Wraps vkCmdWriteTimestamp2 (sync2).
void writeTimestamp(VkCommandBuffer cmd, const QueryPool& pool,
                    VkPipelineStageFlags2 stage, std::uint32_t query);

void resetQueries(VkCommandBuffer cmd, VkQueryPool pool,
                  std::uint32_t first, std::uint32_t count);

void writeTimestamp(VkCommandBuffer cmd, VkQueryPool pool,
                    VkPipelineStageFlags2 stage, std::uint32_t query);

} // namespace vksdl
