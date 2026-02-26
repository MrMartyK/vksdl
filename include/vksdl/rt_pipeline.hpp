#pragma once

#include <vksdl/error.hpp>
#include <vksdl/pipeline.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace vksdl {

// Common stage flag combinations for RT push constants and descriptor bindings.
inline constexpr VkShaderStageFlags kAllRtStages =
    VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_MISS_BIT_KHR |
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

inline constexpr VkShaderStageFlags kRtHitStages =
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

class Device;
class PipelineCache;

// Thread safety: thread-confined.
class RayTracingPipelineBuilder {
public:
    explicit RayTracingPipelineBuilder(const Device& device);

    // Path-based shader stages (builder loads, creates, and destroys modules).
    RayTracingPipelineBuilder& rayGenShader(const std::filesystem::path& spvPath);
    RayTracingPipelineBuilder& missShader(const std::filesystem::path& spvPath);
    RayTracingPipelineBuilder& closestHitShader(const std::filesystem::path& spvPath);
    RayTracingPipelineBuilder& anyHitShader(const std::filesystem::path& spvPath);
    RayTracingPipelineBuilder& intersectionShader(const std::filesystem::path& spvPath);

    // Module-based (user owns VkShaderModule lifetime).
    RayTracingPipelineBuilder& rayGenModule(VkShaderModule module);
    RayTracingPipelineBuilder& missModule(VkShaderModule module);
    RayTracingPipelineBuilder& closestHitModule(VkShaderModule module);
    RayTracingPipelineBuilder& anyHitModule(VkShaderModule module);
    RayTracingPipelineBuilder& intersectionModule(VkShaderModule module);

    // Hit groups. Indices refer to the stage index within each category
    // (0 = first closestHit, 1 = second, etc.). Not the global stage index.
    // If no explicit hit groups are added, auto-grouping creates one
    // TRIANGLES_HIT_GROUP per closest-hit shader.
    RayTracingPipelineBuilder& addTrianglesHitGroup(
        std::uint32_t closestHitIndex,
        std::uint32_t anyHitIndex = VK_SHADER_UNUSED_KHR);
    RayTracingPipelineBuilder& addProceduralHitGroup(
        std::uint32_t intersectionIndex,
        std::uint32_t closestHitIndex,
        std::uint32_t anyHitIndex = VK_SHADER_UNUSED_KHR);

    RayTracingPipelineBuilder& maxRecursionDepth(std::uint32_t depth);

    // Pipeline cache.
    RayTracingPipelineBuilder& cache(const PipelineCache& c);
    RayTracingPipelineBuilder& cache(VkPipelineCache c);

    // Pipeline layout.
    RayTracingPipelineBuilder& pushConstantRange(VkPushConstantRange range);
    RayTracingPipelineBuilder& descriptorSetLayout(VkDescriptorSetLayout layout);
    RayTracingPipelineBuilder& pipelineLayout(VkPipelineLayout layout);

    // Convenience: deduces size from struct type, single range at offset 0.
    template<typename T>
    RayTracingPipelineBuilder& pushConstants(VkShaderStageFlags stages) {
        return pushConstantRange({stages, 0, static_cast<std::uint32_t>(sizeof(T))});
    }
    RayTracingPipelineBuilder& pushConstants(VkShaderStageFlags stages, std::uint32_t size) {
        return pushConstantRange({stages, 0, size});
    }

    [[nodiscard]] Result<Pipeline> build();

private:
    enum class StageType { RayGen, Miss, ClosestHit, AnyHit, Intersection };

    struct StageEntry {
        StageType              type;
        std::filesystem::path  path;
        VkShaderModule         module = VK_NULL_HANDLE;
    };

    struct HitGroupEntry {
        bool          procedural       = false;
        std::uint32_t closestHitIndex  = VK_SHADER_UNUSED_KHR;
        std::uint32_t anyHitIndex      = VK_SHADER_UNUSED_KHR;
        std::uint32_t intersectionIndex = VK_SHADER_UNUSED_KHR;
    };

    [[nodiscard]] Result<VkShaderModule> createModule(
        const std::vector<std::uint32_t>& code) const;

    VkDevice device_ = VK_NULL_HANDLE;

    std::vector<StageEntry>    stages_;
    std::vector<HitGroupEntry> hitGroups_;
    std::uint32_t              maxRecursion_ = 1;

    std::vector<VkPushConstantRange>   pushConstantRanges_;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts_;
    VkPipelineLayout                   externalLayout_ = VK_NULL_HANDLE;
    VkPipelineCache                    cache_ = VK_NULL_HANDLE;
};

} // namespace vksdl
