#pragma once

#include <vksdl/pipeline.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <type_traits>
#include <vector>

namespace vksdl {

class Device;
class PipelineCache;

class ComputePipelineBuilder {
  public:
    explicit ComputePipelineBuilder(const Device& device);

    // Path-based: builder loads SPIR-V, creates module, destroys after build.
    ComputePipelineBuilder& shader(const std::filesystem::path& spvPath);

    // Module-based: user owns the VkShaderModule lifetime.
    ComputePipelineBuilder& shaderModule(VkShaderModule module);

    // Specialization constants.
    template <typename T>
    ComputePipelineBuilder& specConstant(std::uint32_t constantId, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "specConstant requires a trivially copyable type");
        VkSpecializationMapEntry entry{};
        entry.constantID = constantId;
        entry.offset = static_cast<std::uint32_t>(specData_.size());
        entry.size = sizeof(T);
        specEntries_.push_back(entry);
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
        specData_.insert(specData_.end(), bytes, bytes + sizeof(T));
        return *this;
    }
    ComputePipelineBuilder& specialize(const VkSpecializationInfo& info);

    ComputePipelineBuilder& cache(const PipelineCache& c);
    ComputePipelineBuilder& cache(VkPipelineCache c);

    ComputePipelineBuilder& pushConstantRange(VkPushConstantRange range);
    ComputePipelineBuilder& descriptorSetLayout(VkDescriptorSetLayout layout);
    ComputePipelineBuilder& pipelineLayout(VkPipelineLayout layout);

    // Convenience: deduces size from struct type, single range at offset 0.
    template <typename T> ComputePipelineBuilder& pushConstants(VkShaderStageFlags stages) {
        return pushConstantRange({stages, 0, static_cast<std::uint32_t>(sizeof(T))});
    }
    ComputePipelineBuilder& pushConstants(VkShaderStageFlags stages, std::uint32_t size) {
        return pushConstantRange({stages, 0, size});
    }

    // Reflect the compute shader (must use path-based shader) and create
    // descriptor set layouts + push constant ranges automatically.
    ComputePipelineBuilder& reflectDescriptors();

    [[nodiscard]] Result<Pipeline> build();

  private:
    [[nodiscard]] Result<VkShaderModule> createModule(const std::vector<std::uint32_t>& code) const;

    VkDevice device_ = VK_NULL_HANDLE;

    std::filesystem::path shaderPath_;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;

    std::vector<VkPushConstantRange> pushConstantRanges_;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts_;
    VkPipelineLayout externalLayout_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    bool reflect_ = false;

    std::vector<VkSpecializationMapEntry> specEntries_;
    std::vector<std::uint8_t> specData_;
    std::optional<VkSpecializationInfo> externalSpecInfo_;
};

} // namespace vksdl
