#include <vksdl/device.hpp>
#include <vksdl/mesh_pipeline.hpp>
#include <vksdl/pipeline_cache.hpp>
#include <vksdl/swapchain.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vksdl {

MeshPipelineBuilder::MeshPipelineBuilder(const Device& device) : device_(device.vkDevice()) {}

MeshPipelineBuilder& MeshPipelineBuilder::taskShader(const std::filesystem::path& spvPath) {
    taskPath_ = spvPath;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::meshShader(const std::filesystem::path& spvPath) {
    meshPath_ = spvPath;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::fragmentShader(const std::filesystem::path& spvPath) {
    fragPath_ = spvPath;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::taskModule(VkShaderModule module) {
    taskModule_ = module;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::meshModule(VkShaderModule module) {
    meshModule_ = module;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::fragmentModule(VkShaderModule module) {
    fragModule_ = module;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::colorFormat(const Swapchain& swapchain) {
    colorFormat_ = swapchain.format();
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::colorFormat(VkFormat format) {
    colorFormat_ = format;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::depthFormat(VkFormat format) {
    depthFormat_ = format;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::cullBack() {
    cullMode_ = VK_CULL_MODE_BACK_BIT;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::cullFront() {
    cullMode_ = VK_CULL_MODE_FRONT_BIT;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::wireframe() {
    polygonMode_ = VK_POLYGON_MODE_LINE;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::clockwise() {
    frontFace_ = VK_FRONT_FACE_CLOCKWISE;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::enableBlending() {
    enableBlending_ = true;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::polygonMode(VkPolygonMode m) {
    polygonMode_ = m;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::cullMode(VkCullModeFlags m) {
    cullMode_ = m;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::frontFace(VkFrontFace f) {
    frontFace_ = f;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::samples(VkSampleCountFlagBits s) {
    samples_ = s;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::depthCompareOp(VkCompareOp op) {
    depthCompareOp_ = op;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::dynamicState(VkDynamicState state) {
    extraDynamicStates_.push_back(state);
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::dynamicCullMode() {
    return dynamicState(VK_DYNAMIC_STATE_CULL_MODE);
}

MeshPipelineBuilder& MeshPipelineBuilder::dynamicDepthTest() {
    dynamicState(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    dynamicState(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    return dynamicState(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);
}

MeshPipelineBuilder& MeshPipelineBuilder::dynamicFrontFace() {
    return dynamicState(VK_DYNAMIC_STATE_FRONT_FACE);
}

MeshPipelineBuilder& MeshPipelineBuilder::specialize(const VkSpecializationInfo& info) {
    externalSpecInfo_ = info;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::cache(const PipelineCache& c) {
    cache_ = c.vkPipelineCache();
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::pushConstantRange(VkPushConstantRange range) {
#ifndef NDEBUG
    if (range.offset + range.size > 128) {
        std::fprintf(stderr,
                     "[vksdl perf] push constant range exceeds 128 bytes (%u bytes). "
                     "The Vulkan spec only guarantees 128 bytes. "
                     "Consider using a uniform buffer instead.\n",
                     range.offset + range.size);
    }
#endif
    pushConstantRanges_.push_back(range);
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::descriptorSetLayout(VkDescriptorSetLayout layout) {
    descriptorSetLayouts_.push_back(layout);
    return *this;
}

MeshPipelineBuilder& MeshPipelineBuilder::pipelineLayout(VkPipelineLayout layout) {
    externalLayout_ = layout;
    return *this;
}

Result<VkShaderModule>
MeshPipelineBuilder::createModule(const std::vector<std::uint32_t>& code) const {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size() * sizeof(std::uint32_t);
    ci.pCode = code.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(device_, &ci, nullptr, &module);
    if (vr != VK_SUCCESS) {
        return Error{"create mesh shader module", static_cast<std::int32_t>(vr),
                     "vkCreateShaderModule failed"};
    }
    return module;
}

Result<Pipeline> MeshPipelineBuilder::build() {
    bool hasMeshShader = !meshPath_.empty() || meshModule_ != VK_NULL_HANDLE;
    bool hasFragShader = !fragPath_.empty() || fragModule_ != VK_NULL_HANDLE;
    bool hasTaskShader = !taskPath_.empty() || taskModule_ != VK_NULL_HANDLE;

    if (!hasMeshShader) {
        return Error{"create mesh pipeline", 0,
                     "no mesh shader set -- call meshShader(path) or meshModule(module)"};
    }
    if (!hasFragShader) {
        return Error{
            "create mesh pipeline", 0,
            "no fragment shader set -- call fragmentShader(path) or fragmentModule(module)"};
    }
    if (colorFormat_ == VK_FORMAT_UNDEFINED) {
        return Error{"create mesh pipeline", 0,
                     "no color format set -- call colorFormat(swapchain) or colorFormat(VkFormat)"};
    }

    VkShaderModule taskMod = taskModule_;
    VkShaderModule meshMod = meshModule_;
    VkShaderModule fragMod = fragModule_;
    bool createdTask = false;
    bool createdMesh = false;
    bool createdFrag = false;

    auto destroyModules = [&]() {
        if (createdTask && taskMod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, taskMod, nullptr);
        }
        if (createdMesh && meshMod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, meshMod, nullptr);
        }
        if (createdFrag && fragMod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, fragMod, nullptr);
        }
    };

    if (hasTaskShader && taskMod == VK_NULL_HANDLE) {
        auto code = readSpv(taskPath_);
        if (!code.ok()) {
            return std::move(code).error();
        }
        auto mod = createModule(code.value());
        if (!mod.ok()) {
            return std::move(mod).error();
        }
        taskMod = mod.value();
        createdTask = true;
    }

    if (meshMod == VK_NULL_HANDLE) {
        auto code = readSpv(meshPath_);
        if (!code.ok()) {
            destroyModules();
            return std::move(code).error();
        }
        auto mod = createModule(code.value());
        if (!mod.ok()) {
            destroyModules();
            return std::move(mod).error();
        }
        meshMod = mod.value();
        createdMesh = true;
    }

    if (fragMod == VK_NULL_HANDLE) {
        auto code = readSpv(fragPath_);
        if (!code.ok()) {
            destroyModules();
            return std::move(code).error();
        }
        auto mod = createModule(code.value());
        if (!mod.ok()) {
            destroyModules();
            return std::move(mod).error();
        }
        fragMod = mod.value();
        createdFrag = true;
    }

    Pipeline p;
    p.device_ = device_;

    if (externalLayout_ != VK_NULL_HANDLE) {
        p.layout_ = externalLayout_;
        p.ownsLayout_ = false;
    } else {
        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount = static_cast<std::uint32_t>(descriptorSetLayouts_.size());
        layoutCI.pSetLayouts =
            descriptorSetLayouts_.empty() ? nullptr : descriptorSetLayouts_.data();
        layoutCI.pushConstantRangeCount = static_cast<std::uint32_t>(pushConstantRanges_.size());
        layoutCI.pPushConstantRanges =
            pushConstantRanges_.empty() ? nullptr : pushConstantRanges_.data();

        VkResult vr = vkCreatePipelineLayout(device_, &layoutCI, nullptr, &p.layout_);
        if (vr != VK_SUCCESS) {
            destroyModules();
            return Error{"create mesh pipeline layout", static_cast<std::int32_t>(vr),
                         "vkCreatePipelineLayout failed"};
        }
        p.ownsLayout_ = true;
    }

    // Build specialization info from accumulated entries or external info.
    VkSpecializationInfo builtSpecInfo{};
    const VkSpecializationInfo* pSpecInfo = nullptr;
    if (externalSpecInfo_) {
        pSpecInfo = &*externalSpecInfo_;
    } else if (!specEntries_.empty()) {
        builtSpecInfo.mapEntryCount = static_cast<std::uint32_t>(specEntries_.size());
        builtSpecInfo.pMapEntries = specEntries_.data();
        builtSpecInfo.dataSize = specData_.size();
        builtSpecInfo.pData = specData_.data();
        pSpecInfo = &builtSpecInfo;
    }

    // Build shader stage array. Task is optional, so stage count is 2 or 3.
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(3);

    if (hasTaskShader) {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        stage.module = taskMod;
        stage.pName = "main";
        stage.pSpecializationInfo = pSpecInfo;
        stages.push_back(stage);
    }

    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        stage.module = meshMod;
        stage.pName = "main";
        stage.pSpecializationInfo = pSpecInfo;
        stages.push_back(stage);
    }

    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage.module = fragMod;
        stage.pName = "main";
        stage.pSpecializationInfo = pSpecInfo;
        stages.push_back(stage);
    }

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = polygonMode_;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = cullMode_;
    rasterizer.frontFace = frontFace_;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = samples_;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (enableBlending_) {
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    if (depthFormat_ != VK_FORMAT_UNDEFINED) {
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = depthCompareOp_;
    }

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    for (auto s : extraDynamicStates_) {
        if (s != VK_DYNAMIC_STATE_VIEWPORT && s != VK_DYNAMIC_STATE_SCISSOR) {
            dynamicStates.push_back(s);
        }
    }

    VkPipelineDynamicStateCreateInfo dynamicStateCI{};
    dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCI.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicStateCI.pDynamicStates = dynamicStates.data();

    // Dynamic rendering (Vulkan 1.3 core): no VkRenderPass.
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat_;
    if (depthFormat_ != VK_FORMAT_UNDEFINED) {
        renderingInfo.depthAttachmentFormat = depthFormat_;
    }

    // Stage feedback array matches stage count (2 or 3).
    std::uint32_t stageCount = static_cast<std::uint32_t>(stages.size());
    VkPipelineCreationFeedback pipelineFeedback{};
    std::vector<VkPipelineCreationFeedback> stageFeedbacks(stageCount);

    VkPipelineCreationFeedbackCreateInfo feedbackCI{};
    feedbackCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
    feedbackCI.pNext = &renderingInfo;
    feedbackCI.pPipelineCreationFeedback = &pipelineFeedback;
    feedbackCI.pipelineStageCreationFeedbackCount = stageCount;
    feedbackCI.pPipelineStageCreationFeedbacks = stageFeedbacks.data();

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext = &feedbackCI;
    pipelineCI.stageCount = stageCount;
    pipelineCI.pStages = stages.data();
    // Mesh shaders have no vertex input or input assembly state.
    pipelineCI.pVertexInputState = nullptr;
    pipelineCI.pInputAssemblyState = nullptr;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pRasterizationState = &rasterizer;
    pipelineCI.pMultisampleState = &multisampling;
    pipelineCI.pDepthStencilState = &depthStencil;
    pipelineCI.pColorBlendState = &colorBlend;
    pipelineCI.pDynamicState = &dynamicStateCI;
    pipelineCI.layout = p.layout_;
    pipelineCI.renderPass = VK_NULL_HANDLE;

    VkResult vr = vkCreateGraphicsPipelines(device_, cache_, 1, &pipelineCI, nullptr, &p.pipeline_);

    destroyModules();

    if (vr != VK_SUCCESS) {
        if (p.ownsLayout_ && p.layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, p.layout_, nullptr);
            p.layout_ = VK_NULL_HANDLE;
        }
        return Error{"create mesh pipeline", static_cast<std::int32_t>(vr),
                     "vkCreateGraphicsPipelines failed for mesh pipeline"};
    }

    p.bindPoint_ = VK_PIPELINE_BIND_POINT_GRAPHICS;
    for (const auto& r : pushConstantRanges_) {
        p.pcStages_ |= r.stageFlags;
        auto end = r.offset + r.size;
        if (end > p.pcSize_)
            p.pcSize_ = end;
    }

    PipelineStats stats;
    stats.valid = (pipelineFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
    stats.cacheHit = (pipelineFeedback.flags &
                      VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
    stats.durationMs = static_cast<double>(pipelineFeedback.duration) / 1'000'000.0;

    // Populate per-stage feedback in the order stages were added.
    if (hasTaskShader) {
        StageFeedback sf;
        sf.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        sf.valid = (stageFeedbacks[0].flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
        sf.cacheHit = (stageFeedbacks[0].flags &
                       VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
        sf.durationMs = static_cast<double>(stageFeedbacks[0].duration) / 1'000'000.0;
        stats.stages.push_back(sf);
    }

    std::uint32_t meshIdx = hasTaskShader ? 1u : 0u;
    std::uint32_t fragIdx = meshIdx + 1u;

    {
        StageFeedback sf;
        sf.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        sf.valid = (stageFeedbacks[meshIdx].flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
        sf.cacheHit = (stageFeedbacks[meshIdx].flags &
                       VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
        sf.durationMs = static_cast<double>(stageFeedbacks[meshIdx].duration) / 1'000'000.0;
        stats.stages.push_back(sf);
    }

    {
        StageFeedback sf;
        sf.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        sf.valid = (stageFeedbacks[fragIdx].flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
        sf.cacheHit = (stageFeedbacks[fragIdx].flags &
                       VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
        sf.durationMs = static_cast<double>(stageFeedbacks[fragIdx].duration) / 1'000'000.0;
        stats.stages.push_back(sf);
    }

    p.stats_ = std::move(stats);

    return p;
}

} // namespace vksdl
