#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/shader_reflect.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <vector>

namespace vksdl {

class Device;
class DescriptorSet;
class PipelineCache;
class PipelineCompiler;
class Swapchain;

// Per-stage creation feedback from VkPipelineCreationFeedback (Vulkan 1.3 core).
struct StageFeedback {
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL;
    bool     valid      = false;
    bool     cacheHit   = false;
    double   durationMs = 0.0;
};

// Whole-pipeline creation feedback. Always populated after a successful build().
struct PipelineStats {
    bool     valid      = false;
    bool     cacheHit   = false;
    double   durationMs = 0.0;
    std::vector<StageFeedback> stages;
};

// Load a SPIR-V shader binary from disk.
// Returns the code as uint32_t words suitable for VkShaderModuleCreateInfo.
[[nodiscard]] Result<std::vector<std::uint32_t>> readSpv(const std::filesystem::path& path);

// RAII graphics pipeline. Owns VkPipeline and (optionally) VkPipelineLayout.
// Destroys pipeline before layout (pipeline references the layout).
//
// Thread safety: immutable after construction.
class Pipeline {
public:
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    [[nodiscard]] VkPipeline       native()          const { return pipeline_; }
    [[nodiscard]] VkPipeline       vkPipeline()      const { return native(); }
    [[nodiscard]] VkPipelineLayout vkPipelineLayout() const { return layout_; }

    // Descriptor set layouts owned by shader reflection.
    // Empty when reflectDescriptors() was not used.
    [[nodiscard]] const std::vector<VkDescriptorSetLayout>& reflectedSetLayouts() const {
        return ownedSetLayouts_;
    }
    [[nodiscard]] const ReflectedLayout* reflectedLayout() const {
        return reflectedLayout_ ? &*reflectedLayout_ : nullptr;
    }

    // Bind this pipeline to a command buffer.
    void bind(VkCommandBuffer cmd) const;

    // Bind this pipeline and a descriptor set.
    void bind(VkCommandBuffer cmd, const DescriptorSet& ds) const;
    void bind(VkCommandBuffer cmd, VkDescriptorSet ds) const;

    // Push constants. Uses the stage flags and size recorded at build time.
    // Only valid when the pipeline was built with a push constant range.
    template<typename T>
    void pushConstants(VkCommandBuffer cmd, const T& data) const {
        pushConstants(cmd, &data, static_cast<std::uint32_t>(sizeof(T)));
    }
    void pushConstants(VkCommandBuffer cmd, const void* data,
                       std::uint32_t size) const;

    // Pipeline creation feedback (Vulkan 1.3 core).
    // Returns nullptr only when pipeline was not built through a builder.
    [[nodiscard]] const PipelineStats* feedback() const {
        return stats_ ? &*stats_ : nullptr;
    }

    // Extended dynamic state helpers (Vulkan 1.3 core).
    // Use with dynamicCullMode() / dynamicDepthTest() / dynamicTopology() /
    // dynamicFrontFace() on PipelineBuilder.
    static void setCullMode(VkCommandBuffer cmd, VkCullModeFlags mode);
    static void setDepthTest(VkCommandBuffer cmd, bool enable,
                             bool write = true,
                             VkCompareOp op = VK_COMPARE_OP_LESS_OR_EQUAL);
    static void setTopology(VkCommandBuffer cmd, VkPrimitiveTopology t);
    static void setFrontFace(VkCommandBuffer cmd, VkFrontFace f);

private:
    friend class PipelineBuilder;
    friend class ComputePipelineBuilder;
    friend class RayTracingPipelineBuilder;
    friend class PipelineCompiler;
    Pipeline() = default;

    VkDevice            device_     = VK_NULL_HANDLE;
    VkPipeline          pipeline_   = VK_NULL_HANDLE;
    VkPipelineLayout    layout_     = VK_NULL_HANDLE;
    bool                ownsLayout_ = true;
    VkPipelineBindPoint bindPoint_  = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkShaderStageFlags  pcStages_   = 0;
    std::uint32_t       pcSize_     = 0;
    // Descriptor set layouts owned by reflection (destroyed before layout).
    std::vector<VkDescriptorSetLayout> ownedSetLayouts_;
    std::optional<ReflectedLayout> reflectedLayout_;
    // Pipeline creation feedback (populated by builders).
    std::optional<PipelineStats> stats_;
};

class PipelineBuilder {
public:
    explicit PipelineBuilder(const Device& device);

    // Path-based: builder loads, creates, and destroys modules.
    PipelineBuilder& vertexShader(const std::filesystem::path& spvPath);
    PipelineBuilder& fragmentShader(const std::filesystem::path& spvPath);
    PipelineBuilder& simpleColorPipeline(const std::filesystem::path& vertSpvPath,
                                         const std::filesystem::path& fragSpvPath,
                                         const Swapchain& swapchain);

    // Module-based: user owns the VkShaderModule lifetime.
    PipelineBuilder& vertexModule(VkShaderModule module);
    PipelineBuilder& fragmentModule(VkShaderModule module);

    // Dynamic rendering (Vulkan 1.3, no VkRenderPass).
    PipelineBuilder& colorFormat(const Swapchain& swapchain);
    PipelineBuilder& colorFormat(VkFormat format);
    PipelineBuilder& depthFormat(VkFormat format);

    // Vertex input -- default: none (geometry hardcoded in shader).
    PipelineBuilder& vertexBinding(std::uint32_t binding,
                                   std::uint32_t stride,
                                   VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX);
    PipelineBuilder& vertexAttribute(std::uint32_t location,
                                     std::uint32_t binding,
                                     VkFormat format,
                                     std::uint32_t offset);

    PipelineBuilder& cullBack();          // cullMode = VK_CULL_MODE_BACK_BIT
    PipelineBuilder& cullFront();         // cullMode = VK_CULL_MODE_FRONT_BIT
    PipelineBuilder& wireframe();         // polygonMode = VK_POLYGON_MODE_LINE
    PipelineBuilder& clockwise();          // frontFace = VK_FRONT_FACE_CLOCKWISE
    PipelineBuilder& enableBlending();    // standard alpha blending

    PipelineBuilder& topology(VkPrimitiveTopology t);
    PipelineBuilder& polygonMode(VkPolygonMode m);
    PipelineBuilder& cullMode(VkCullModeFlags m);
    PipelineBuilder& frontFace(VkFrontFace f);
    PipelineBuilder& samples(VkSampleCountFlagBits s);
    PipelineBuilder& depthCompareOp(VkCompareOp op);
    PipelineBuilder& dynamicState(VkDynamicState state);

    // Extended dynamic state (Vulkan 1.3 core).
    PipelineBuilder& dynamicCullMode();
    PipelineBuilder& dynamicDepthTest();
    PipelineBuilder& dynamicTopology();
    PipelineBuilder& dynamicFrontFace();

    PipelineBuilder& cache(const PipelineCache& c);
    PipelineBuilder& cache(VkPipelineCache c);

    PipelineBuilder& pushConstantRange(VkPushConstantRange range);
    PipelineBuilder& descriptorSetLayout(VkDescriptorSetLayout layout);
    PipelineBuilder& pipelineLayout(VkPipelineLayout layout);

    // When set, build() reflects both shader stages (must use path-based shaders),
    // merges the reflected layouts, and creates descriptor set layouts + push constant
    // ranges automatically. Manual descriptorSetLayout() / pushConstantRange() calls
    // are still used as escape hatches when reflection is not desired.
    PipelineBuilder& reflectDescriptors();

    // Convenience: deduces size from struct type, single range at offset 0.
    template<typename T>
    PipelineBuilder& pushConstants(VkShaderStageFlags stages) {
        return pushConstantRange({stages, 0, static_cast<std::uint32_t>(sizeof(T))});
    }
    PipelineBuilder& pushConstants(VkShaderStageFlags stages, std::uint32_t size) {
        return pushConstantRange({stages, 0, size});
    }

    [[nodiscard]] Result<Pipeline> build();

    // Build with extra VkPipelineCreateFlags. Escape hatch for cache-only
    // probes (FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) and GPL library flags.
    // Const because it only reads builder state and fills stack-local create infos.
    [[nodiscard]] Result<Pipeline> buildWithFlags(VkPipelineCreateFlags flags) const;

private:
    friend class PipelineCompiler;

    [[nodiscard]] Result<VkShaderModule> createModule(
        const std::vector<std::uint32_t>& code) const;

    VkDevice device_ = VK_NULL_HANDLE;

    // Shaders: either a path (loaded in build) or a pre-created module.
    std::filesystem::path vertPath_;
    std::filesystem::path fragPath_;
    VkShaderModule        vertModule_ = VK_NULL_HANDLE;
    VkShaderModule        fragModule_ = VK_NULL_HANDLE;

    // Dynamic rendering
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat_  = VK_FORMAT_UNDEFINED;

    // Vertex input
    std::vector<VkVertexInputBindingDescription>   vertexBindings_;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes_;

    // Input assembly
    VkPrimitiveTopology topology_ = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Rasterization
    VkPolygonMode   polygonMode_ = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode_    = VK_CULL_MODE_NONE;  // beginner-safe default
    VkFrontFace     frontFace_   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // Multisampling
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;

    // Color blending
    bool enableBlending_ = false;

    // Depth compare op (used when depthFormat_ is set)
    VkCompareOp depthCompareOp_ = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Extra dynamic states (viewport + scissor are always included)
    std::vector<VkDynamicState> extraDynamicStates_;

    // Pipeline layout config
    std::vector<VkPushConstantRange>   pushConstantRanges_;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts_;
    VkPipelineLayout                   externalLayout_ = VK_NULL_HANDLE;
    VkPipelineCache                    cache_ = VK_NULL_HANDLE;
    bool                               reflect_ = false;
};

} // namespace vksdl
