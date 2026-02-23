#pragma once

#include <vksdl/graph/resource.hpp>
#include <vksdl/graph/resource_state.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace vksdl::graph {
struct ResolvedRendering;
struct ResolvedDescriptors;
}

namespace vksdl::graph {

// Per-resource layout info populated by RenderGraph::execute().
struct PassResourceLayout {
    ResourceHandle handle;
    VkImageLayout  layout;
};

// Override entry accumulated by PassContext, applied by the graph after
// the pass callback returns.
struct StateOverride {
    ResourceHandle   handle;
    ResourceState    state;
    SubresourceRange range = {0, VK_REMAINING_MIP_LEVELS,
                               0, VK_REMAINING_ARRAY_LAYERS};
    bool             fullResource = true;
};

// Provided to the pass record callback.
// Resolves resource handles to Vulkan objects and provides escape hatches
// for manual state overrides.
class PassContext {
public:
    [[nodiscard]] VkImage     vkImage(ResourceHandle h) const;
    [[nodiscard]] VkImageView vkImageView(ResourceHandle h) const;
    [[nodiscard]] VkBuffer    vkBuffer(ResourceHandle h) const;
    [[nodiscard]] VkDeviceSize bufferSize(ResourceHandle h) const;

    [[nodiscard]] VkFormat    imageFormat(ResourceHandle h) const;
    [[nodiscard]] VkExtent2D  imageExtent(ResourceHandle h) const;

    // The image layout determined by the barrier compiler for this resource
    // in the current pass. Only valid for image resources declared in this
    // pass's setup function.
    [[nodiscard]] VkImageLayout imageLayout(ResourceHandle h) const;

    // Build a VkDescriptorImageInfo from the resource's current view, layout,
    // and the given sampler.
    [[nodiscard]] VkDescriptorImageInfo imageInfo(ResourceHandle h,
                                                   VkSampler sampler) const;

    // Build a VkDescriptorBufferInfo for a buffer resource.
    [[nodiscard]] VkDescriptorBufferInfo bufferInfo(ResourceHandle h) const;

    // Bind the pipeline passed to the Layer 2 addPass() overload.
    void bindPipeline(VkCommandBuffer cmd);

    // Bind all auto-allocated descriptor sets. Skips VK_NULL_HANDLE entries
    // (unmanaged sets the user must bind manually).
    void bindDescriptors(VkCommandBuffer cmd);

    // True if this pass was declared with the Layer 2 addPass() overload.
    [[nodiscard]] bool hasPipeline() const { return descriptors_ != nullptr; }

    // Escape hatch: retrieve the auto-allocated descriptor set at a given
    // set index. Returns VK_NULL_HANDLE for unmanaged sets.
    [[nodiscard]] VkDescriptorSet descriptorSet(std::uint32_t setIndex) const;

    // Begin dynamic rendering using pre-resolved render targets from the
    // setup lambda's setColorTarget() / setDepthTarget() declarations.
    // Also sets viewport and scissor to the render area extent.
    // If the pass was declared with Layer 2, also binds pipeline + descriptors.
    // Only valid when the pass declared render targets (hasRenderTargets()).
    void beginRendering(VkCommandBuffer cmd);

    // End dynamic rendering. Must be called before the record callback returns
    // if beginRendering() was called.
    void endRendering(VkCommandBuffer cmd);

    // True if the pass declared Layer 1 render targets.
    [[nodiscard]] bool hasRenderTargets() const { return rendering_ != nullptr; }

    // True if beginRendering() was called but endRendering() has not yet.
    [[nodiscard]] bool renderingActive() const { return renderingBegun_; }

    // Override the tracked state for a resource. Used when a pass performs
    // manual transitions or when the actual final state differs from what
    // was declared. Applied after the callback returns.
    void assumeState(ResourceHandle h, const ResourceState& state);
    void assumeState(ResourceHandle h, const ResourceState& state,
                     const SubresourceRange& range);

    // Discard contents of an image resource. Sets layout to UNDEFINED and
    // clears all tracking state. Use only when the resource will be fully
    // overwritten by its next use.
    void discardContents(ResourceHandle h);

    [[nodiscard]] const std::vector<StateOverride>& overrides() const {
        return overrides_;
    }

private:
    friend class RenderGraph;

    PassContext(const std::vector<ResourceEntry>& resources,
               std::vector<PassResourceLayout> layouts,
               const ResolvedRendering* rendering,
               const ResolvedDescriptors* descriptors = nullptr)
        : resources_(&resources), layouts_(std::move(layouts)),
          rendering_(rendering), descriptors_(descriptors) {}

    const std::vector<ResourceEntry>* resources_;
    std::vector<StateOverride>        overrides_;
    std::vector<PassResourceLayout>   layouts_;
    const ResolvedRendering*          rendering_    = nullptr;
    const ResolvedDescriptors*        descriptors_  = nullptr;
    bool                              renderingBegun_ = false;
};

} // namespace vksdl::graph
