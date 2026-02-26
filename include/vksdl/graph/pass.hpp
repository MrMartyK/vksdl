#pragma once

#include <vksdl/graph/resource.hpp>
#include <vksdl/graph/resource_state.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vksdl { struct ReflectedLayout; }

namespace vksdl::graph {

// Layer 2: bind map entry. Maps a reflected shader binding name to a graph
// resource handle, with optional per-binding sampler override.
struct BindEntry {
    ResourceHandle handle;
    VkSampler      samplerOverride = VK_NULL_HANDLE; // null = use pass default
};

// Three-layer API: L0 raw access, L1 render targets, L2 pipeline-aware auto-bind.

class PassContext;

enum class PassType : std::uint8_t {
    Graphics,   // Runs on graphics queue.
    Compute,    // Runs inline on graphics queue; reserved for future async compute.
    Transfer,   // Runs inline on graphics queue; reserved for future async DMA.
};

enum class AccessType : std::uint8_t {
    Read,
    Write,
    ReadWrite,
};

enum class LoadOp : std::uint8_t {
    Clear,      // Clear to a specified value before the pass.
    Load,       // Preserve previous contents.
    DontCare,   // Contents undefined -- use when writing every pixel.
};

enum class DepthWrite : std::uint8_t {
    Enabled,    // Depth test + write. Layout: DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    Disabled,   // Depth test only.   Layout: DEPTH_STENCIL_READ_ONLY_OPTIMAL.
};

struct ColorTargetDecl {
    std::uint32_t     index  = 0;
    ResourceHandle    handle;
    LoadOp            loadOp     = LoadOp::Clear;
    VkClearColorValue clearValue = {{0.0f, 0.0f, 0.0f, 0.0f}};
};

struct DepthTargetDecl {
    ResourceHandle handle;
    LoadOp         loadOp       = LoadOp::Clear;
    DepthWrite     depthWrite   = DepthWrite::Enabled;
    float          clearDepth   = 1.0f;
    std::uint32_t  clearStencil = 0;
};

// A single resource access declaration within a pass.
struct ResourceAccess {
    ResourceHandle   handle;
    AccessType       access       = AccessType::Read;
    ResourceState    desiredState;
    SubresourceRange subresourceRange = {0, VK_REMAINING_MIP_LEVELS,
                                         0, VK_REMAINING_ARRAY_LAYERS};
};

// Callback types.
using RecordFn = std::function<void(class PassContext&, VkCommandBuffer)>;
using SetupFn  = std::function<void(class PassBuilder&)>;

// Internal pass representation after declaration.
struct PassDecl {
    std::string                 name;
    PassType                    type = PassType::Graphics;
    std::vector<ResourceAccess> accesses;
    RecordFn                    recordFn;

    std::vector<ColorTargetDecl>      colorTargets;
    std::optional<DepthTargetDecl>    depthTarget;

    VkPipeline                                 pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout                           pipelineLayout = VK_NULL_HANDLE;
    const ReflectedLayout*                     reflection     = nullptr; // must outlive compile+execute
    VkSampler                                  defaultSampler = VK_NULL_HANDLE;
    std::unordered_map<std::string, BindEntry> bindMap;
};

// Fluent builder for declaring resource accesses within a pass.
// Created by RenderGraph::addPass() setup callback.
//
// Thread safety: thread-confined.
class PassBuilder {
public:
    // Declare a color render target. Implies writeColorAttachment().
    PassBuilder& setColorTarget(std::uint32_t index, ResourceHandle h,
                                LoadOp loadOp = LoadOp::Clear);

    // Overload with explicit clear value.
    PassBuilder& setColorTarget(std::uint32_t index, ResourceHandle h,
                                LoadOp loadOp, VkClearColorValue clearValue);

    // Declare a depth render target.
    // DepthWrite::Enabled  -> implies writeDepthAttachment() (read+write).
    // DepthWrite::Disabled -> implies readDepthAttachment()  (read-only).
    PassBuilder& setDepthTarget(ResourceHandle h,
                                LoadOp loadOp = LoadOp::Clear,
                                DepthWrite depthWrite = DepthWrite::Enabled);

    // Overload with explicit clear depth value.
    PassBuilder& setDepthTarget(ResourceHandle h,
                                LoadOp loadOp, DepthWrite depthWrite,
                                float clearDepth, std::uint32_t clearStencil = 0);

    // Through sampler, not storage.
    PassBuilder& sampleImage(ResourceHandle h,
                             SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                                                        0, VK_REMAINING_ARRAY_LAYERS});

    PassBuilder& readStorageImage(ResourceHandle h,
                                  SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                                                             0, VK_REMAINING_ARRAY_LAYERS});

    PassBuilder& readTransferSrc(ResourceHandle h,
                                  SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                                                             0, VK_REMAINING_ARRAY_LAYERS});

    // Input attachment read (for future subpass merging).
    PassBuilder& readInputAttachment(ResourceHandle h);

    // Depth testing without writes.
    PassBuilder& readDepthAttachment(ResourceHandle h);

    PassBuilder& writeColorAttachment(ResourceHandle h);
    PassBuilder& writeDepthAttachment(ResourceHandle h);

    PassBuilder& writeStorageImage(ResourceHandle h,
                                   SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                                                              0, VK_REMAINING_ARRAY_LAYERS});

    PassBuilder& writeTransferDst(ResourceHandle h,
                                   SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                                                              0, VK_REMAINING_ARRAY_LAYERS});

    PassBuilder& readUniformBuffer(ResourceHandle h);
    PassBuilder& readStorageBuffer(ResourceHandle h);
    PassBuilder& readVertexBuffer(ResourceHandle h);
    PassBuilder& readIndexBuffer(ResourceHandle h);
    PassBuilder& readIndirectBuffer(ResourceHandle h);
    PassBuilder& readTransferSrcBuffer(ResourceHandle h);

    PassBuilder& writeStorageBuffer(ResourceHandle h);
    PassBuilder& writeTransferDstBuffer(ResourceHandle h);

    // Set a default sampler used for all COMBINED_IMAGE_SAMPLER / SAMPLED_IMAGE
    // bindings in this pass (unless overridden per-binding).
    PassBuilder& setSampler(VkSampler sampler);

    // Map a reflected shader binding name to a graph resource handle.
    // The descriptor type is inferred from reflection metadata.
    PassBuilder& bind(std::string_view name, ResourceHandle h);

    // Map with per-binding sampler override.
    PassBuilder& bind(std::string_view name, ResourceHandle h,
                      VkSampler samplerOverride);

    PassBuilder& access(ResourceHandle h, AccessType type,
                        ResourceState desiredState,
                        SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                                                   0, VK_REMAINING_ARRAY_LAYERS});

private:
    friend class RenderGraph;

    explicit PassBuilder(PassType type) : type_(type) {}

    // Derive pipeline stage from PassType for shader-accessible resources.
    [[nodiscard]] VkPipelineStageFlags2 shaderStage() const;

    PassType                    type_;
    std::vector<ResourceAccess> accesses_;

    // Layer 1 state, moved into PassDecl by addPass().
    std::vector<ColorTargetDecl>   colorTargets_;
    std::optional<DepthTargetDecl> depthTarget_;

    // Layer 2 state, moved into PassDecl by addPass().
    VkSampler                                  defaultSampler_ = VK_NULL_HANDLE;
    std::unordered_map<std::string, BindEntry> bindMap_;
};

} // namespace vksdl::graph
