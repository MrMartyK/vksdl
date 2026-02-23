#include <vksdl/graph/pass_context.hpp>
#include <vksdl/graph/render_graph.hpp> // ResolvedRendering

#include <cassert>

namespace vksdl::graph {

VkImage PassContext::vkImage(ResourceHandle h) const {
    assert(h.valid() && h.index < resources_->size());
    return (*resources_)[h.index].vkImage;
}

VkImageView PassContext::vkImageView(ResourceHandle h) const {
    assert(h.valid() && h.index < resources_->size());
    return (*resources_)[h.index].vkImageView;
}

VkBuffer PassContext::vkBuffer(ResourceHandle h) const {
    assert(h.valid() && h.index < resources_->size());
    return (*resources_)[h.index].vkBuffer;
}

VkDeviceSize PassContext::bufferSize(ResourceHandle h) const {
    assert(h.valid() && h.index < resources_->size());
    return (*resources_)[h.index].bufferSize;
}

VkFormat PassContext::imageFormat(ResourceHandle h) const {
    assert(h.valid() && h.index < resources_->size());
    assert((*resources_)[h.index].kind == ResourceKind::Image
           && "imageFormat() called on a buffer resource");
    return (*resources_)[h.index].imageDesc.format;
}

VkExtent2D PassContext::imageExtent(ResourceHandle h) const {
    assert(h.valid() && h.index < resources_->size());
    assert((*resources_)[h.index].kind == ResourceKind::Image
           && "imageExtent() called on a buffer resource");
    const auto& desc = (*resources_)[h.index].imageDesc;
    return {desc.width, desc.height};
}

VkImageLayout PassContext::imageLayout(ResourceHandle h) const {
    assert(h.valid());
    for (const auto& pl : layouts_) {
        if (pl.handle == h) return pl.layout;
    }
    assert(false && "imageLayout: resource not declared in this pass");
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

VkDescriptorImageInfo PassContext::imageInfo(ResourceHandle h,
                                              VkSampler sampler) const {
    VkDescriptorImageInfo info{};
    info.sampler     = sampler;
    info.imageView   = vkImageView(h);
    info.imageLayout = imageLayout(h);
    return info;
}

VkDescriptorBufferInfo PassContext::bufferInfo(ResourceHandle h) const {
    VkDescriptorBufferInfo info{};
    info.buffer = vkBuffer(h);
    info.offset = 0;
    info.range  = bufferSize(h);
    return info;
}

void PassContext::bindPipeline(VkCommandBuffer cmd) {
    assert(descriptors_ && "bindPipeline: pass has no Layer 2 pipeline");
    vkCmdBindPipeline(cmd, descriptors_->bindPoint, descriptors_->pipeline);
}

void PassContext::bindDescriptors(VkCommandBuffer cmd) {
    assert(descriptors_ && "bindDescriptors: pass has no Layer 2 descriptors");
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(descriptors_->sets.size()); ++i) {
        VkDescriptorSet set = descriptors_->sets[i];
        if (set == VK_NULL_HANDLE) continue;
        vkCmdBindDescriptorSets(cmd, descriptors_->bindPoint,
                                descriptors_->pipelineLayout,
                                i, 1, &set, 0, nullptr);
    }
}

VkDescriptorSet PassContext::descriptorSet(std::uint32_t setIndex) const {
    if (!descriptors_ || setIndex >= descriptors_->sets.size())
        return VK_NULL_HANDLE;
    return descriptors_->sets[setIndex];
}

void PassContext::beginRendering(VkCommandBuffer cmd) {
    assert(rendering_ && "beginRendering: pass has no render targets declared");
    assert(!renderingBegun_ && "beginRendering: already active");

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.offset    = {0, 0};
    ri.renderArea.extent    = rendering_->renderArea;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = static_cast<std::uint32_t>(
                                  rendering_->colorAttachments.size());
    ri.pColorAttachments    = rendering_->colorAttachments.empty()
                            ? nullptr
                            : rendering_->colorAttachments.data();
    ri.pDepthAttachment     = rendering_->hasDepth
                            ? &rendering_->depthAttachment
                            : nullptr;

    vkCmdBeginRendering(cmd, &ri);
    renderingBegun_ = true;

    VkViewport vp{};
    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = static_cast<float>(rendering_->renderArea.width);
    vp.height   = static_cast<float>(rendering_->renderArea.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = rendering_->renderArea;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Layer 2: auto-bind pipeline and descriptors after viewport/scissor.
    if (descriptors_) {
        bindPipeline(cmd);
        bindDescriptors(cmd);
    }
}

void PassContext::endRendering(VkCommandBuffer cmd) {
    assert(renderingBegun_ && "endRendering: no active rendering");
    vkCmdEndRendering(cmd);
    renderingBegun_ = false;
}

void PassContext::assumeState(ResourceHandle h, const ResourceState& state) {
    overrides_.push_back({h, state, {}, true});
}

void PassContext::assumeState(ResourceHandle h, const ResourceState& state,
                               const SubresourceRange& range) {
    overrides_.push_back({h, state, range, false});
}

void PassContext::discardContents(ResourceHandle h) {
    ResourceState discarded{};
    discarded.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    discarded.queueFamily   = VK_QUEUE_FAMILY_IGNORED;
    overrides_.push_back({h, discarded, {}, true});
}

} // namespace vksdl::graph
