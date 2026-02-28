#include <vksdl/device.hpp>
#include <vksdl/query_pool.hpp>

namespace vksdl {

QueryPool::~QueryPool() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

QueryPool::QueryPool(QueryPool&& o) noexcept
    : device_(o.device_), pool_(o.pool_), type_(o.type_), count_(o.count_) {
    o.device_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.count_ = 0;
}

QueryPool& QueryPool::operator=(QueryPool&& o) noexcept {
    if (this != &o) {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, pool_, nullptr);
        }
        device_ = o.device_;
        pool_ = o.pool_;
        type_ = o.type_;
        count_ = o.count_;
        o.device_ = VK_NULL_HANDLE;
        o.pool_ = VK_NULL_HANDLE;
        o.count_ = 0;
    }
    return *this;
}

Result<QueryPool> QueryPool::create(const Device& device, VkQueryType type, std::uint32_t count) {
    if (count == 0) {
        return Error{"create query pool", 0, "query count must be > 0"};
    }

    VkQueryPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = type;
    ci.queryCount = count;

    QueryPool qp;
    qp.device_ = device.vkDevice();
    qp.type_ = type;
    qp.count_ = count;

    VkResult vr = vkCreateQueryPool(qp.device_, &ci, nullptr, &qp.pool_);
    if (vr != VK_SUCCESS) {
        return Error{"create query pool", static_cast<std::int32_t>(vr),
                     "vkCreateQueryPool failed"};
    }

    return qp;
}

Result<std::vector<std::uint64_t>> QueryPool::getResults(std::uint32_t first,
                                                         std::uint32_t resultCount,
                                                         VkQueryResultFlags flags) const {

    std::vector<std::uint64_t> results(resultCount);

    VkResult vr = vkGetQueryPoolResults(device_, pool_, first, resultCount,
                                        resultCount * sizeof(std::uint64_t), results.data(),
                                        sizeof(std::uint64_t), flags | VK_QUERY_RESULT_64_BIT);

    if (vr != VK_SUCCESS && vr != VK_NOT_READY) {
        return Error{"get query pool results", static_cast<std::int32_t>(vr),
                     "vkGetQueryPoolResults failed"};
    }

    return results;
}

void resetQueries(VkCommandBuffer cmd, const QueryPool& pool, std::uint32_t first,
                  std::uint32_t count) {
    vkCmdResetQueryPool(cmd, pool.vkQueryPool(), first, count);
}

void writeTimestamp(VkCommandBuffer cmd, const QueryPool& pool, VkPipelineStageFlags2 stage,
                    std::uint32_t query) {
    vkCmdWriteTimestamp2(cmd, stage, pool.vkQueryPool(), query);
}

void resetQueries(VkCommandBuffer cmd, VkQueryPool pool, std::uint32_t first, std::uint32_t count) {
    vkCmdResetQueryPool(cmd, pool, first, count);
}

void writeTimestamp(VkCommandBuffer cmd, VkQueryPool pool, VkPipelineStageFlags2 stage,
                    std::uint32_t query) {
    vkCmdWriteTimestamp2(cmd, stage, pool, query);
}

} // namespace vksdl
