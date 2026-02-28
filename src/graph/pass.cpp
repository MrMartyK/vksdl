#include <vksdl/graph/pass.hpp>

namespace vksdl::graph {

VkPipelineStageFlags2 PassBuilder::shaderStage() const {
    switch (type_) {
    case PassType::Compute:
        return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    case PassType::Graphics:
        return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    case PassType::Transfer:
        return VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    }
    return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
}

PassBuilder& PassBuilder::sampleImage(ResourceHandle h, SubresourceRange range) {
    ResourceState state{};
    state.lastWriteStage = shaderStage();
    state.readAccessSinceWrite = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    accesses_.push_back({h, AccessType::Read, state, range});
    return *this;
}

PassBuilder& PassBuilder::readStorageImage(ResourceHandle h, SubresourceRange range) {
    ResourceState state{};
    state.lastWriteStage = shaderStage();
    state.readAccessSinceWrite = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
    accesses_.push_back({h, AccessType::Read, state, range});
    return *this;
}

PassBuilder& PassBuilder::readTransferSrc(ResourceHandle h, SubresourceRange range) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_TRANSFER_READ_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    accesses_.push_back({h, AccessType::Read, state, range});
    return *this;
}

PassBuilder& PassBuilder::readInputAttachment(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::readDepthAttachment(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::writeColorAttachment(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    state.lastWriteAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    accesses_.push_back({h, AccessType::Write, state, {}});
    return *this;
}

PassBuilder& PassBuilder::writeDepthAttachment(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    state.lastWriteAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    accesses_.push_back({h, AccessType::ReadWrite, state, {}});
    return *this;
}

PassBuilder& PassBuilder::writeStorageImage(ResourceHandle h, SubresourceRange range) {
    ResourceState state{};
    state.lastWriteStage = shaderStage();
    state.lastWriteAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
    accesses_.push_back({h, AccessType::Write, state, range});
    return *this;
}

PassBuilder& PassBuilder::writeTransferDst(ResourceHandle h, SubresourceRange range) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    state.lastWriteAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    state.currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    accesses_.push_back({h, AccessType::Write, state, range});
    return *this;
}

PassBuilder& PassBuilder::readUniformBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = shaderStage();
    state.readAccessSinceWrite = VK_ACCESS_2_UNIFORM_READ_BIT;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::readStorageBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = shaderStage();
    state.readAccessSinceWrite = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::readVertexBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::readIndexBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_INDEX_READ_BIT;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::readIndirectBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::readTransferSrcBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    state.readAccessSinceWrite = VK_ACCESS_2_TRANSFER_READ_BIT;
    accesses_.push_back({h, AccessType::Read, state, {}});
    return *this;
}

PassBuilder& PassBuilder::writeStorageBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = shaderStage();
    state.lastWriteAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    accesses_.push_back({h, AccessType::Write, state, {}});
    return *this;
}

PassBuilder& PassBuilder::writeTransferDstBuffer(ResourceHandle h) {
    ResourceState state{};
    state.lastWriteStage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    state.lastWriteAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    accesses_.push_back({h, AccessType::Write, state, {}});
    return *this;
}

PassBuilder& PassBuilder::setColorTarget(std::uint32_t index, ResourceHandle h, LoadOp loadOp) {
    writeColorAttachment(h);
    colorTargets_.push_back({index, h, loadOp, {{0.0f, 0.0f, 0.0f, 0.0f}}});
    return *this;
}

PassBuilder& PassBuilder::setColorTarget(std::uint32_t index, ResourceHandle h, LoadOp loadOp,
                                         VkClearColorValue clearValue) {
    writeColorAttachment(h);
    colorTargets_.push_back({index, h, loadOp, clearValue});
    return *this;
}

PassBuilder& PassBuilder::setDepthTarget(ResourceHandle h, LoadOp loadOp, DepthWrite depthWrite) {
    if (depthWrite == DepthWrite::Enabled)
        writeDepthAttachment(h);
    else
        readDepthAttachment(h);
    depthTarget_ = DepthTargetDecl{h, loadOp, depthWrite, 1.0f, 0};
    return *this;
}

PassBuilder& PassBuilder::setDepthTarget(ResourceHandle h, LoadOp loadOp, DepthWrite depthWrite,
                                         float clearDepth, std::uint32_t clearStencil) {
    if (depthWrite == DepthWrite::Enabled)
        writeDepthAttachment(h);
    else
        readDepthAttachment(h);
    depthTarget_ = DepthTargetDecl{h, loadOp, depthWrite, clearDepth, clearStencil};
    return *this;
}

PassBuilder& PassBuilder::setSampler(VkSampler sampler) {
    defaultSampler_ = sampler;
    return *this;
}

PassBuilder& PassBuilder::bind(std::string_view name, ResourceHandle h) {
    bindMap_[std::string(name)] = BindEntry{h, VK_NULL_HANDLE};
    return *this;
}

PassBuilder& PassBuilder::bind(std::string_view name, ResourceHandle h, VkSampler samplerOverride) {
    bindMap_[std::string(name)] = BindEntry{h, samplerOverride};
    return *this;
}

PassBuilder& PassBuilder::access(ResourceHandle h, AccessType type, ResourceState desiredState,
                                 SubresourceRange range) {
    accesses_.push_back({h, type, desiredState, range});
    return *this;
}

} // namespace vksdl::graph
