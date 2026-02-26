#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl {

class Device;
class PipelineCache;

// RAII wrapper for a GPL library part (VkPipeline with
// VK_PIPELINE_CREATE_LIBRARY_BIT_KHR). Movable-only.
// Destroyed when the GplLibrary goes out of scope.
//
// Thread safety: immutable after construction.
class GplLibrary {
public:
    ~GplLibrary();
    GplLibrary(GplLibrary&&) noexcept;
    GplLibrary& operator=(GplLibrary&&) noexcept;
    GplLibrary(const GplLibrary&) = delete;
    GplLibrary& operator=(const GplLibrary&) = delete;

    [[nodiscard]] VkPipeline vkPipeline() const { return pipeline_; }

private:
    friend class GplVertexInputBuilder;
    friend class GplPreRasterizationBuilder;
    friend class GplFragmentShaderBuilder;
    friend class GplFragmentOutputBuilder;
    GplLibrary() = default;

    VkDevice   device_   = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// GPL library builders
//
// Each builder creates one pipeline library part. The four parts are:
//   1. Vertex Input Interface (VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT)
//   2. Pre-Rasterization Shaders (VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT)
//   3. Fragment Shader (VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT)
//   4. Fragment Output Interface (VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT)
//
// pMultisampleState split (the #1 source of GPL bugs):
//
// Both Fragment Shader and Fragment Output Interface libraries must provide
// a non-null pMultisampleState. Each gets a separate VkPipelineMultisampleStateCreateInfo
// with only its relevant fields populated:
//
// | Field                | Fragment Shader | Fragment Output Interface |
// |----------------------|-----------------|---------------------------|
// | rasterizationSamples | zero/ignored    | OWNED                     |
// | sampleShadingEnable  | OWNED           | VK_FALSE                  |
// | minSampleShading     | OWNED           | zero/ignored              |
// | pSampleMask          | null            | OWNED                     |
// | alphaToCoverageEnable| VK_FALSE        | OWNED                     |
// | alphaToOneEnable     | VK_FALSE        | OWNED                     |
//
// DXVK got this wrong initially and it caused driver crashes on AMD.

// 1. Vertex Input Interface
class GplVertexInputBuilder {
public:
    explicit GplVertexInputBuilder(const Device& device);

    GplVertexInputBuilder& vertexBinding(std::uint32_t binding,
                                         std::uint32_t stride,
                                         VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX);
    GplVertexInputBuilder& vertexAttribute(std::uint32_t location,
                                           std::uint32_t binding,
                                           VkFormat format,
                                           std::uint32_t offset);
    GplVertexInputBuilder& topology(VkPrimitiveTopology t);
    GplVertexInputBuilder& primitiveRestart(bool enable);
    GplVertexInputBuilder& cache(VkPipelineCache c);
    GplVertexInputBuilder& dynamicState(VkDynamicState state);

    [[nodiscard]] Result<GplLibrary> build() const;

private:
    VkDevice device_ = VK_NULL_HANDLE;

    std::vector<VkVertexInputBindingDescription>   vertexBindings_;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes_;
    VkPrimitiveTopology topology_          = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool                primitiveRestart_  = false;
    VkPipelineCache     cache_             = VK_NULL_HANDLE;
    std::vector<VkDynamicState> extraDynamicStates_;
};

// 2. Pre-Rasterization Shaders
class GplPreRasterizationBuilder {
public:
    explicit GplPreRasterizationBuilder(const Device& device);

    GplPreRasterizationBuilder& vertexModule(VkShaderModule module);
    GplPreRasterizationBuilder& polygonMode(VkPolygonMode m);
    GplPreRasterizationBuilder& cullMode(VkCullModeFlags m);
    GplPreRasterizationBuilder& frontFace(VkFrontFace f);
    GplPreRasterizationBuilder& lineWidth(float w);
    GplPreRasterizationBuilder& pipelineLayout(VkPipelineLayout layout);
    GplPreRasterizationBuilder& cache(VkPipelineCache c);
    GplPreRasterizationBuilder& dynamicState(VkDynamicState state);
    // Dynamic rendering viewMask (0 for single-view).
    GplPreRasterizationBuilder& viewMask(std::uint32_t mask);

    [[nodiscard]] Result<GplLibrary> build() const;

private:
    VkDevice           device_      = VK_NULL_HANDLE;
    VkShaderModule     vertModule_  = VK_NULL_HANDLE;
    VkPolygonMode      polygonMode_ = VK_POLYGON_MODE_FILL;
    VkCullModeFlags    cullMode_    = VK_CULL_MODE_NONE;
    VkFrontFace        frontFace_   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    float              lineWidth_   = 1.0f;
    VkPipelineLayout   layout_      = VK_NULL_HANDLE;
    VkPipelineCache    cache_       = VK_NULL_HANDLE;
    std::uint32_t      viewMask_    = 0;
    std::vector<VkDynamicState> extraDynamicStates_;
};

// 3. Fragment Shader
class GplFragmentShaderBuilder {
public:
    explicit GplFragmentShaderBuilder(const Device& device);

    GplFragmentShaderBuilder& fragmentModule(VkShaderModule module);
    GplFragmentShaderBuilder& depthTest(bool enable, bool write = true,
                                        VkCompareOp op = VK_COMPARE_OP_LESS_OR_EQUAL);
    // Fragment shader owns sampleShadingEnable and minSampleShading.
    GplFragmentShaderBuilder& sampleShading(bool enable, float min = 1.0f);
    GplFragmentShaderBuilder& pipelineLayout(VkPipelineLayout layout);
    GplFragmentShaderBuilder& cache(VkPipelineCache c);
    GplFragmentShaderBuilder& dynamicState(VkDynamicState state);

    [[nodiscard]] Result<GplLibrary> build() const;

private:
    VkDevice         device_             = VK_NULL_HANDLE;
    VkShaderModule   fragModule_         = VK_NULL_HANDLE;
    bool             depthTestEnable_    = false;
    bool             depthWriteEnable_   = false;
    VkCompareOp      depthCompareOp_     = VK_COMPARE_OP_LESS_OR_EQUAL;
    bool             sampleShadingEnable_ = false;
    float            minSampleShading_   = 1.0f;
    VkPipelineLayout layout_             = VK_NULL_HANDLE;
    VkPipelineCache  cache_              = VK_NULL_HANDLE;
    std::vector<VkDynamicState> extraDynamicStates_;
};

// 4. Fragment Output Interface
class GplFragmentOutputBuilder {
public:
    explicit GplFragmentOutputBuilder(const Device& device);

    GplFragmentOutputBuilder& colorFormat(VkFormat format);
    GplFragmentOutputBuilder& depthFormat(VkFormat format);
    GplFragmentOutputBuilder& samples(VkSampleCountFlagBits s);
    GplFragmentOutputBuilder& enableBlending();
    GplFragmentOutputBuilder& alphaToCoverage(bool enable);
    GplFragmentOutputBuilder& alphaToOne(bool enable);
    GplFragmentOutputBuilder& cache(VkPipelineCache c);
    GplFragmentOutputBuilder& dynamicState(VkDynamicState state);

    [[nodiscard]] Result<GplLibrary> build() const;

private:
    VkDevice              device_              = VK_NULL_HANDLE;
    VkFormat              colorFormat_         = VK_FORMAT_UNDEFINED;
    VkFormat              depthFormat_         = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_             = VK_SAMPLE_COUNT_1_BIT;
    bool                  enableBlending_      = false;
    bool                  alphaToCoverage_     = false;
    bool                  alphaToOne_          = false;
    VkPipelineCache       cache_               = VK_NULL_HANDLE;
    std::vector<VkDynamicState> extraDynamicStates_;
};

// Link four GPL library parts into a final pipeline.
// optimized=false: fast path (no VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT).
// optimized=true: full link-time optimization (slower, better runtime performance).
[[nodiscard]] Result<VkPipeline> linkGplPipeline(
    const Device& device,
    const GplLibrary& vi, const GplLibrary& pr,
    const GplLibrary& fs, const GplLibrary& fo,
    VkPipelineLayout layout,
    VkPipelineCache cache = VK_NULL_HANDLE,
    bool optimized = false);

} // namespace vksdl
