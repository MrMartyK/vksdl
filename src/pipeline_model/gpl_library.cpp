#include <vksdl/pipeline_model/gpl_library.hpp>
#include <vksdl/device.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vksdl {

GplLibrary::~GplLibrary() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
}

GplLibrary::GplLibrary(GplLibrary&& o) noexcept
    : device_(o.device_), pipeline_(o.pipeline_) {
    o.device_   = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
}

GplLibrary& GplLibrary::operator=(GplLibrary&& o) noexcept {
    if (this != &o) {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        device_   = o.device_;
        pipeline_ = o.pipeline_;
        o.device_   = VK_NULL_HANDLE;
        o.pipeline_ = VK_NULL_HANDLE;
    }
    return *this;
}

GplVertexInputBuilder::GplVertexInputBuilder(const Device& device)
    : device_(device.vkDevice()) {}

GplVertexInputBuilder& GplVertexInputBuilder::vertexBinding(
    std::uint32_t binding, std::uint32_t stride, VkVertexInputRate inputRate) {
    vertexBindings_.push_back({binding, stride, inputRate});
    return *this;
}

GplVertexInputBuilder& GplVertexInputBuilder::vertexAttribute(
    std::uint32_t location, std::uint32_t binding,
    VkFormat format, std::uint32_t offset) {
    vertexAttributes_.push_back({location, binding, format, offset});
    return *this;
}

GplVertexInputBuilder& GplVertexInputBuilder::topology(VkPrimitiveTopology t) {
    topology_ = t;
    return *this;
}

GplVertexInputBuilder& GplVertexInputBuilder::primitiveRestart(bool enable) {
    primitiveRestart_ = enable;
    return *this;
}

GplVertexInputBuilder& GplVertexInputBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

GplVertexInputBuilder& GplVertexInputBuilder::dynamicState(VkDynamicState state) {
    extraDynamicStates_.push_back(state);
    return *this;
}

Result<GplLibrary> GplVertexInputBuilder::build() const {
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount =
        static_cast<std::uint32_t>(vertexBindings_.size());
    vertexInput.pVertexBindingDescriptions =
        vertexBindings_.empty() ? nullptr : vertexBindings_.data();
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(vertexAttributes_.size());
    vertexInput.pVertexAttributeDescriptions =
        vertexAttributes_.empty() ? nullptr : vertexAttributes_.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology_;
    inputAssembly.primitiveRestartEnable = primitiveRestart_ ? VK_TRUE : VK_FALSE;

    std::vector<VkDynamicState> dynamicStates = extraDynamicStates_;

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.empty() ? nullptr : dynamicStates.data();

    VkGraphicsPipelineLibraryCreateInfoEXT libCI{};
    libCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
    libCI.flags = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext = &libCI;
    pipelineCI.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
                     | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    pipelineCI.pVertexInputState   = &vertexInput;
    pipelineCI.pInputAssemblyState = &inputAssembly;
    pipelineCI.pDynamicState       = &dynamicState;

    GplLibrary lib;
    lib.device_ = device_;

    VkResult vr = vkCreateGraphicsPipelines(
        device_, cache_, 1, &pipelineCI, nullptr, &lib.pipeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create GPL vertex input library",
                     static_cast<std::int32_t>(vr),
                     "vkCreateGraphicsPipelines failed for VERTEX_INPUT_INTERFACE"};
    }

    return lib;
}

GplPreRasterizationBuilder::GplPreRasterizationBuilder(const Device& device)
    : device_(device.vkDevice()) {}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::vertexModule(VkShaderModule module) {
    vertModule_ = module;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::polygonMode(VkPolygonMode m) {
    polygonMode_ = m;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::cullMode(VkCullModeFlags m) {
    cullMode_ = m;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::frontFace(VkFrontFace f) {
    frontFace_ = f;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::lineWidth(float w) {
    lineWidth_ = w;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::pipelineLayout(VkPipelineLayout layout) {
    layout_ = layout;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::dynamicState(VkDynamicState state) {
    extraDynamicStates_.push_back(state);
    return *this;
}

GplPreRasterizationBuilder& GplPreRasterizationBuilder::viewMask(std::uint32_t mask) {
    viewMask_ = mask;
    return *this;
}

Result<GplLibrary> GplPreRasterizationBuilder::build() const {
    if (vertModule_ == VK_NULL_HANDLE) {
        return Error{"create GPL pre-rasterization library", 0,
                     "no vertex shader module set"};
    }
    if (layout_ == VK_NULL_HANDLE) {
        return Error{"create GPL pre-rasterization library", 0,
                     "no pipeline layout set -- required for pre-rasterization"};
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stage.module = vertModule_;
    stage.pName  = "main";

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = polygonMode_;
    rasterizer.lineWidth   = lineWidth_;
    rasterizer.cullMode    = cullMode_;
    rasterizer.frontFace   = frontFace_;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    for (auto s : extraDynamicStates_) {
        if (s != VK_DYNAMIC_STATE_VIEWPORT && s != VK_DYNAMIC_STATE_SCISSOR) {
            dynamicStates.push_back(s);
        }
    }

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType    = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.viewMask = viewMask_;

    VkGraphicsPipelineLibraryCreateInfoEXT libCI{};
    libCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
    libCI.pNext = &renderingInfo;
    libCI.flags = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext = &libCI;
    pipelineCI.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
                     | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    pipelineCI.stageCount          = 1;
    pipelineCI.pStages             = &stage;
    pipelineCI.pViewportState      = &viewportState;
    pipelineCI.pRasterizationState = &rasterizer;
    pipelineCI.pDynamicState       = &dynamicState;
    pipelineCI.layout              = layout_;

    GplLibrary lib;
    lib.device_ = device_;

    VkResult vr = vkCreateGraphicsPipelines(
        device_, cache_, 1, &pipelineCI, nullptr, &lib.pipeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create GPL pre-rasterization library",
                     static_cast<std::int32_t>(vr),
                     "vkCreateGraphicsPipelines failed for PRE_RASTERIZATION_SHADERS"};
    }

    return lib;
}

GplFragmentShaderBuilder::GplFragmentShaderBuilder(const Device& device)
    : device_(device.vkDevice()) {}

GplFragmentShaderBuilder& GplFragmentShaderBuilder::fragmentModule(VkShaderModule module) {
    fragModule_ = module;
    return *this;
}

GplFragmentShaderBuilder& GplFragmentShaderBuilder::depthTest(
    bool enable, bool write, VkCompareOp op) {
    depthTestEnable_  = enable;
    depthWriteEnable_ = write;
    depthCompareOp_   = op;
    return *this;
}

GplFragmentShaderBuilder& GplFragmentShaderBuilder::sampleShading(
    bool enable, float min) {
    sampleShadingEnable_ = enable;
    minSampleShading_    = min;
    return *this;
}

GplFragmentShaderBuilder& GplFragmentShaderBuilder::pipelineLayout(VkPipelineLayout layout) {
    layout_ = layout;
    return *this;
}

GplFragmentShaderBuilder& GplFragmentShaderBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

GplFragmentShaderBuilder& GplFragmentShaderBuilder::dynamicState(VkDynamicState state) {
    extraDynamicStates_.push_back(state);
    return *this;
}

Result<GplLibrary> GplFragmentShaderBuilder::build() const {
    if (fragModule_ == VK_NULL_HANDLE) {
        return Error{"create GPL fragment shader library", 0,
                     "no fragment shader module set"};
    }
    if (layout_ == VK_NULL_HANDLE) {
        return Error{"create GPL fragment shader library", 0,
                     "no pipeline layout set -- required for fragment shader"};
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stage.module = fragModule_;
    stage.pName  = "main";

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = depthTestEnable_  ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = depthWriteEnable_ ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp   = depthCompareOp_;

    // Fragment shader owns sampleShadingEnable + minSampleShading only.
    // rasterizationSamples is zeroed (owned by fragment output).
    // pSampleMask, alphaToCoverage, alphaToOne are zeroed (owned by fragment output).
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // placeholder, not owned
    multisampling.sampleShadingEnable  = sampleShadingEnable_ ? VK_TRUE : VK_FALSE;
    multisampling.minSampleShading     = minSampleShading_;

    std::vector<VkDynamicState> dynamicStates = extraDynamicStates_;

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.empty() ? nullptr : dynamicStates.data();

    VkGraphicsPipelineLibraryCreateInfoEXT libCI{};
    libCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
    libCI.flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext = &libCI;
    pipelineCI.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
                     | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    pipelineCI.stageCount          = 1;
    pipelineCI.pStages             = &stage;
    pipelineCI.pDepthStencilState  = &depthStencil;
    pipelineCI.pMultisampleState   = &multisampling;
    pipelineCI.pDynamicState       = &dynamicState;
    pipelineCI.layout              = layout_;

    GplLibrary lib;
    lib.device_ = device_;

    VkResult vr = vkCreateGraphicsPipelines(
        device_, cache_, 1, &pipelineCI, nullptr, &lib.pipeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create GPL fragment shader library",
                     static_cast<std::int32_t>(vr),
                     "vkCreateGraphicsPipelines failed for FRAGMENT_SHADER"};
    }

    return lib;
}

GplFragmentOutputBuilder::GplFragmentOutputBuilder(const Device& device)
    : device_(device.vkDevice()) {}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::colorFormat(VkFormat format) {
    colorFormat_ = format;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::depthFormat(VkFormat format) {
    depthFormat_ = format;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::samples(VkSampleCountFlagBits s) {
    samples_ = s;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::enableBlending() {
    enableBlending_ = true;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::alphaToCoverage(bool enable) {
    alphaToCoverage_ = enable;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::alphaToOne(bool enable) {
    alphaToOne_ = enable;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

GplFragmentOutputBuilder& GplFragmentOutputBuilder::dynamicState(VkDynamicState state) {
    extraDynamicStates_.push_back(state);
    return *this;
}

Result<GplLibrary> GplFragmentOutputBuilder::build() const {
    if (colorFormat_ == VK_FORMAT_UNDEFINED) {
        return Error{"create GPL fragment output library", 0,
                     "no color format set"};
    }

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (enableBlending_) {
        colorBlendAttachment.blendEnable         = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &colorBlendAttachment;

    // Fragment output owns rasterizationSamples, pSampleMask,
    // alphaToCoverage, alphaToOne.
    // sampleShadingEnable + minSampleShading are zeroed (owned by fragment shader).
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = samples_;
    multisampling.alphaToCoverageEnable = alphaToCoverage_ ? VK_TRUE : VK_FALSE;
    multisampling.alphaToOneEnable      = alphaToOne_ ? VK_TRUE : VK_FALSE;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat_;
    if (depthFormat_ != VK_FORMAT_UNDEFINED) {
        renderingInfo.depthAttachmentFormat = depthFormat_;
    }

    std::vector<VkDynamicState> dynamicStates = extraDynamicStates_;

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.empty() ? nullptr : dynamicStates.data();

    VkGraphicsPipelineLibraryCreateInfoEXT libCI{};
    libCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
    libCI.pNext = &renderingInfo;
    libCI.flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext = &libCI;
    pipelineCI.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
                     | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    pipelineCI.pColorBlendState  = &colorBlend;
    pipelineCI.pMultisampleState = &multisampling;
    pipelineCI.pDynamicState     = &dynamicState;

    GplLibrary lib;
    lib.device_ = device_;

    VkResult vr = vkCreateGraphicsPipelines(
        device_, cache_, 1, &pipelineCI, nullptr, &lib.pipeline_);
    if (vr != VK_SUCCESS) {
        return Error{"create GPL fragment output library",
                     static_cast<std::int32_t>(vr),
                     "vkCreateGraphicsPipelines failed for FRAGMENT_OUTPUT_INTERFACE"};
    }

    return lib;
}

Result<VkPipeline> linkGplPipeline(
    const Device& device,
    const GplLibrary& vi, const GplLibrary& pr,
    const GplLibrary& fs, const GplLibrary& fo,
    VkPipelineLayout layout,
    VkPipelineCache cache,
    bool optimized) {

    VkPipeline libraries[4] = {
        vi.vkPipeline(), pr.vkPipeline(),
        fs.vkPipeline(), fo.vkPipeline()
    };

    VkPipelineLibraryCreateInfoKHR libInfo{};
    libInfo.sType        = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
    libInfo.libraryCount = 4;
    libInfo.pLibraries   = libraries;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType  = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext  = &libInfo;
    pipelineCI.layout = layout;
    pipelineCI.flags  = 0;
    if (optimized) {
        pipelineCI.flags |= VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
    }

    VkDevice vkDev = device.vkDevice();
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult vr = vkCreateGraphicsPipelines(
        vkDev, cache, 1, &pipelineCI, nullptr, &pipeline);
    if (vr != VK_SUCCESS) {
        std::string msg = "vkCreateGraphicsPipelines failed during GPL linking";
        if (optimized) msg += " (optimized)";
        return Error{"link GPL pipeline", static_cast<std::int32_t>(vr), msg};
    }

    return pipeline;
}

} // namespace vksdl
