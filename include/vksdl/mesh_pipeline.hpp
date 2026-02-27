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
class Swapchain;

// Builder for VK_EXT_mesh_shader graphics pipelines.
//
// Mesh shaders generate vertices and primitives programmatically, so there is
// no vertex input state or input assembly state. The pipeline consists of:
//   - task shader (optional): VK_SHADER_STAGE_TASK_BIT_EXT
//   - mesh shader (required): VK_SHADER_STAGE_MESH_BIT_EXT
//   - fragment shader (required): VK_SHADER_STAGE_FRAGMENT_BIT
//
// Produces the same Pipeline RAII type as PipelineBuilder.
//
// Thread safety: builder is not thread-safe during construction. The resulting
// Pipeline is immutable after construction.
class MeshPipelineBuilder {
  public:
    explicit MeshPipelineBuilder(const Device& device);

    // Path-based: builder loads SPIR-V, creates module, destroys after build.
    MeshPipelineBuilder& taskShader(const std::filesystem::path& spvPath);
    MeshPipelineBuilder& meshShader(const std::filesystem::path& spvPath);
    MeshPipelineBuilder& fragmentShader(const std::filesystem::path& spvPath);

    // Module-based: user owns the VkShaderModule lifetime.
    MeshPipelineBuilder& taskModule(VkShaderModule module);
    MeshPipelineBuilder& meshModule(VkShaderModule module);
    MeshPipelineBuilder& fragmentModule(VkShaderModule module);

    // Dynamic rendering (Vulkan 1.3, no VkRenderPass).
    MeshPipelineBuilder& colorFormat(const Swapchain& swapchain);
    MeshPipelineBuilder& colorFormat(VkFormat format);
    MeshPipelineBuilder& depthFormat(VkFormat format);

    // Rasterization state.
    MeshPipelineBuilder& cullBack();
    MeshPipelineBuilder& cullFront();
    MeshPipelineBuilder& wireframe();
    MeshPipelineBuilder& clockwise();
    MeshPipelineBuilder& enableBlending();

    MeshPipelineBuilder& polygonMode(VkPolygonMode m);
    MeshPipelineBuilder& cullMode(VkCullModeFlags m);
    MeshPipelineBuilder& frontFace(VkFrontFace f);
    MeshPipelineBuilder& samples(VkSampleCountFlagBits s);
    MeshPipelineBuilder& depthCompareOp(VkCompareOp op);
    MeshPipelineBuilder& dynamicState(VkDynamicState state);

    // Extended dynamic state (Vulkan 1.3 core).
    MeshPipelineBuilder& dynamicCullMode();
    MeshPipelineBuilder& dynamicDepthTest();
    MeshPipelineBuilder& dynamicFrontFace();

    // Specialization constants: applied to all active stages.
    template <typename T>
    MeshPipelineBuilder& specConstant(std::uint32_t constantId, const T& value) {
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

    // Escape hatch: raw VkSpecializationInfo for full control.
    MeshPipelineBuilder& specialize(const VkSpecializationInfo& info);

    MeshPipelineBuilder& cache(const PipelineCache& c);
    MeshPipelineBuilder& cache(VkPipelineCache c);

    MeshPipelineBuilder& pushConstantRange(VkPushConstantRange range);
    MeshPipelineBuilder& descriptorSetLayout(VkDescriptorSetLayout layout);
    MeshPipelineBuilder& pipelineLayout(VkPipelineLayout layout);

    // Convenience: deduces size from struct type, single range at offset 0.
    template <typename T> MeshPipelineBuilder& pushConstants(VkShaderStageFlags stages) {
        return pushConstantRange({stages, 0, static_cast<std::uint32_t>(sizeof(T))});
    }
    MeshPipelineBuilder& pushConstants(VkShaderStageFlags stages, std::uint32_t size) {
        return pushConstantRange({stages, 0, size});
    }

    [[nodiscard]] Result<Pipeline> build();

  private:
    [[nodiscard]] Result<VkShaderModule> createModule(const std::vector<std::uint32_t>& code) const;

    VkDevice device_ = VK_NULL_HANDLE;

    // Shaders: path (loaded in build) or pre-created module.
    // Task shader is optional; mesh and fragment are required.
    std::filesystem::path taskPath_;
    std::filesystem::path meshPath_;
    std::filesystem::path fragPath_;
    VkShaderModule taskModule_ = VK_NULL_HANDLE;
    VkShaderModule meshModule_ = VK_NULL_HANDLE;
    VkShaderModule fragModule_ = VK_NULL_HANDLE;

    // Dynamic rendering
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    // Rasterization
    VkPolygonMode polygonMode_ = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode_ = VK_CULL_MODE_NONE;
    VkFrontFace frontFace_ = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // Multisampling
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;

    // Color blending
    bool enableBlending_ = false;

    // Depth compare op (used when depthFormat_ is set)
    VkCompareOp depthCompareOp_ = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Extra dynamic states (viewport + scissor always included)
    std::vector<VkDynamicState> extraDynamicStates_;

    // Pipeline layout config
    std::vector<VkPushConstantRange> pushConstantRanges_;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts_;
    VkPipelineLayout externalLayout_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;

    // Specialization constants
    std::vector<VkSpecializationMapEntry> specEntries_;
    std::vector<std::uint8_t> specData_;
    std::optional<VkSpecializationInfo> externalSpecInfo_;
};

} // namespace vksdl
