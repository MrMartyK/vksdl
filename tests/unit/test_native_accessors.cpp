#include <vksdl/allocator.hpp>
#include <vksdl/bindless_table.hpp>
#include <vksdl/blas.hpp>
#include <vksdl/buffer.hpp>
#include <vksdl/descriptor_set.hpp>
#include <vksdl/device.hpp>
#include <vksdl/image.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/pipeline.hpp>
#include <vksdl/pipeline_cache.hpp>
#include <vksdl/query_pool.hpp>
#include <vksdl/sampler.hpp>
#include <vksdl/surface.hpp>
#include <vksdl/swapchain.hpp>
#include <vksdl/tlas.hpp>

#include <type_traits>

namespace {

template <typename T, typename Handle, typename = void> struct NativeIs : std::false_type {};

template <typename T, typename Handle>
struct NativeIs<T, Handle, std::void_t<decltype(std::declval<const T&>().native())>>
    : std::bool_constant<std::is_same_v<
          std::remove_cv_t<std::remove_reference_t<decltype(std::declval<const T&>().native())>>,
          Handle>> {};

} // namespace

static_assert(NativeIs<vksdl::Allocator, VmaAllocator>::value);
static_assert(NativeIs<vksdl::Instance, VkInstance>::value);
static_assert(NativeIs<vksdl::Surface, VkSurfaceKHR>::value);
static_assert(NativeIs<vksdl::Device, VkDevice>::value);
static_assert(NativeIs<vksdl::Swapchain, VkSwapchainKHR>::value);
static_assert(NativeIs<vksdl::Buffer, VkBuffer>::value);
static_assert(NativeIs<vksdl::Image, VkImage>::value);
static_assert(NativeIs<vksdl::Pipeline, VkPipeline>::value);
static_assert(NativeIs<vksdl::PipelineCache, VkPipelineCache>::value);
static_assert(NativeIs<vksdl::Sampler, VkSampler>::value);
static_assert(NativeIs<vksdl::DescriptorSet, VkDescriptorSet>::value);
static_assert(NativeIs<vksdl::BindlessTable, VkDescriptorSet>::value);
static_assert(NativeIs<vksdl::QueryPool, VkQueryPool>::value);
static_assert(NativeIs<vksdl::Blas, VkAccelerationStructureKHR>::value);
static_assert(NativeIs<vksdl::Tlas, VkAccelerationStructureKHR>::value);

int main() {
    return 0;
}
