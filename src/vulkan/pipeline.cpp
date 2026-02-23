#include <vksdl/pipeline.hpp>
#include <vksdl/descriptor_set.hpp>
#include <vksdl/device.hpp>
#include <vksdl/pipeline_cache.hpp>
#include <vksdl/shader_reflect.hpp>
#include <vksdl/swapchain.hpp>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace vksdl {

Result<std::vector<std::uint32_t>> readSpv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return Error{"read SPIR-V", 0,
            "could not open file: " + path.string()};
    }

    auto pos = file.tellg();
    if (pos <= 0) {
        return Error{"read SPIR-V", 0,
            "file is empty or unreadable: " + path.string()};
    }
    auto size = static_cast<std::size_t>(pos);
    if (size % sizeof(std::uint32_t) != 0) {
        return Error{"read SPIR-V", 0,
            "file size is not a multiple of 4 bytes: " + path.string()};
    }

    std::vector<std::uint32_t> code(size / sizeof(std::uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()),
              static_cast<std::streamsize>(size));

    if (code.size() >= 1 && code[0] != 0x07230203) {
        return Error{"read SPIR-V", 0,
            "invalid SPIR-V magic number in: " + path.string()};
    }

    return code;
}

Pipeline::~Pipeline() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE && ownsLayout_) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
    for (auto dsl : ownedSetLayouts_) {
        vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
    }
}

Pipeline::Pipeline(Pipeline&& o) noexcept
    : device_(o.device_)
    , pipeline_(o.pipeline_)
    , layout_(o.layout_)
    , ownsLayout_(o.ownsLayout_)
    , bindPoint_(o.bindPoint_)
    , pcStages_(o.pcStages_)
    , pcSize_(o.pcSize_)
    , ownedSetLayouts_(std::move(o.ownedSetLayouts_))
    , stats_(std::move(o.stats_)) {
    o.device_   = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
    o.layout_   = VK_NULL_HANDLE;
}

Pipeline& Pipeline::operator=(Pipeline&& o) noexcept {
    if (this != &o) {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE && ownsLayout_) {
            vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
        for (auto dsl : ownedSetLayouts_) {
            vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
        }

        device_     = o.device_;
        pipeline_   = o.pipeline_;
        layout_     = o.layout_;
        ownsLayout_ = o.ownsLayout_;
        bindPoint_  = o.bindPoint_;
        pcStages_   = o.pcStages_;
        pcSize_     = o.pcSize_;
        ownedSetLayouts_ = std::move(o.ownedSetLayouts_);
        stats_      = std::move(o.stats_);

        o.device_   = VK_NULL_HANDLE;
        o.pipeline_ = VK_NULL_HANDLE;
        o.layout_   = VK_NULL_HANDLE;
    }
    return *this;
}

void Pipeline::bind(VkCommandBuffer cmd) const {
    vkCmdBindPipeline(cmd, bindPoint_, pipeline_);
}

void Pipeline::bind(VkCommandBuffer cmd, const DescriptorSet& ds) const {
    bind(cmd, ds.vkDescriptorSet());
}

void Pipeline::bind(VkCommandBuffer cmd, VkDescriptorSet ds) const {
    vkCmdBindPipeline(cmd, bindPoint_, pipeline_);
    vkCmdBindDescriptorSets(cmd, bindPoint_, layout_, 0, 1, &ds, 0, nullptr);
}

void Pipeline::pushConstants(VkCommandBuffer cmd, const void* data,
                             std::uint32_t size) const {
    vkCmdPushConstants(cmd, layout_, pcStages_, 0, size, data);
}

PipelineBuilder::PipelineBuilder(const Device& device)
    : device_(device.vkDevice()) {}

PipelineBuilder& PipelineBuilder::vertexShader(const std::filesystem::path& spvPath) {
    vertPath_ = spvPath;
    return *this;
}

PipelineBuilder& PipelineBuilder::fragmentShader(const std::filesystem::path& spvPath) {
    fragPath_ = spvPath;
    return *this;
}

PipelineBuilder& PipelineBuilder::vertexModule(VkShaderModule module) {
    vertModule_ = module;
    return *this;
}

PipelineBuilder& PipelineBuilder::fragmentModule(VkShaderModule module) {
    fragModule_ = module;
    return *this;
}

PipelineBuilder& PipelineBuilder::colorFormat(const Swapchain& swapchain) {
    colorFormat_ = swapchain.format();
    return *this;
}

PipelineBuilder& PipelineBuilder::colorFormat(VkFormat format) {
    colorFormat_ = format;
    return *this;
}

PipelineBuilder& PipelineBuilder::depthFormat(VkFormat format) {
    depthFormat_ = format;
    return *this;
}

PipelineBuilder& PipelineBuilder::vertexBinding(
    std::uint32_t binding, std::uint32_t stride, VkVertexInputRate inputRate) {
    vertexBindings_.push_back({binding, stride, inputRate});
    return *this;
}

PipelineBuilder& PipelineBuilder::vertexAttribute(
    std::uint32_t location, std::uint32_t binding,
    VkFormat format, std::uint32_t offset) {
    vertexAttributes_.push_back({location, binding, format, offset});
    return *this;
}

PipelineBuilder& PipelineBuilder::cullBack() {
    cullMode_ = VK_CULL_MODE_BACK_BIT;
    return *this;
}

PipelineBuilder& PipelineBuilder::cullFront() {
    cullMode_ = VK_CULL_MODE_FRONT_BIT;
    return *this;
}

PipelineBuilder& PipelineBuilder::wireframe() {
    polygonMode_ = VK_POLYGON_MODE_LINE;
    return *this;
}

PipelineBuilder& PipelineBuilder::clockwise() {
    frontFace_ = VK_FRONT_FACE_CLOCKWISE;
    return *this;
}

PipelineBuilder& PipelineBuilder::enableBlending() {
    enableBlending_ = true;
    return *this;
}

PipelineBuilder& PipelineBuilder::topology(VkPrimitiveTopology t) {
    topology_ = t;
    return *this;
}

PipelineBuilder& PipelineBuilder::polygonMode(VkPolygonMode m) {
    polygonMode_ = m;
    return *this;
}

PipelineBuilder& PipelineBuilder::cullMode(VkCullModeFlags m) {
    cullMode_ = m;
    return *this;
}

PipelineBuilder& PipelineBuilder::frontFace(VkFrontFace f) {
    frontFace_ = f;
    return *this;
}

PipelineBuilder& PipelineBuilder::samples(VkSampleCountFlagBits s) {
    samples_ = s;
    return *this;
}

PipelineBuilder& PipelineBuilder::depthCompareOp(VkCompareOp op) {
    depthCompareOp_ = op;
    return *this;
}

PipelineBuilder& PipelineBuilder::dynamicState(VkDynamicState state) {
    extraDynamicStates_.push_back(state);
    return *this;
}

void Pipeline::setCullMode(VkCommandBuffer cmd, VkCullModeFlags mode) {
    vkCmdSetCullMode(cmd, mode);
}

void Pipeline::setDepthTest(VkCommandBuffer cmd, bool enable, bool write,
                            VkCompareOp op) {
    vkCmdSetDepthTestEnable(cmd, enable ? VK_TRUE : VK_FALSE);
    vkCmdSetDepthWriteEnable(cmd, write ? VK_TRUE : VK_FALSE);
    vkCmdSetDepthCompareOp(cmd, op);
}

void Pipeline::setTopology(VkCommandBuffer cmd, VkPrimitiveTopology t) {
    vkCmdSetPrimitiveTopology(cmd, t);
}

void Pipeline::setFrontFace(VkCommandBuffer cmd, VkFrontFace f) {
    vkCmdSetFrontFace(cmd, f);
}

PipelineBuilder& PipelineBuilder::dynamicCullMode() {
    return dynamicState(VK_DYNAMIC_STATE_CULL_MODE);
}

PipelineBuilder& PipelineBuilder::dynamicDepthTest() {
    dynamicState(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    dynamicState(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    return dynamicState(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);
}

PipelineBuilder& PipelineBuilder::dynamicTopology() {
    return dynamicState(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
}

PipelineBuilder& PipelineBuilder::dynamicFrontFace() {
    return dynamicState(VK_DYNAMIC_STATE_FRONT_FACE);
}

PipelineBuilder& PipelineBuilder::cache(const PipelineCache& c) {
    cache_ = c.vkPipelineCache();
    return *this;
}

PipelineBuilder& PipelineBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

PipelineBuilder& PipelineBuilder::pushConstantRange(VkPushConstantRange range) {
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

PipelineBuilder& PipelineBuilder::descriptorSetLayout(VkDescriptorSetLayout layout) {
    descriptorSetLayouts_.push_back(layout);
    return *this;
}

PipelineBuilder& PipelineBuilder::pipelineLayout(VkPipelineLayout layout) {
    externalLayout_ = layout;
    return *this;
}

PipelineBuilder& PipelineBuilder::reflectDescriptors() {
    reflect_ = true;
    return *this;
}

Result<VkShaderModule> PipelineBuilder::createModule(
    const std::vector<std::uint32_t>& code) const {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size() * sizeof(std::uint32_t);
    ci.pCode    = code.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(device_, &ci, nullptr, &module);
    if (vr != VK_SUCCESS) {
        return Error{"create shader module", static_cast<std::int32_t>(vr),
            "vkCreateShaderModule failed"};
    }
    return module;
}

Result<Pipeline> PipelineBuilder::build() {
    return buildWithFlags(0);
}

Result<Pipeline> PipelineBuilder::buildWithFlags(VkPipelineCreateFlags flags) const {
    bool hasVertShader = !vertPath_.empty() || vertModule_ != VK_NULL_HANDLE;
    bool hasFragShader = !fragPath_.empty() || fragModule_ != VK_NULL_HANDLE;

    if (!hasVertShader) {
        return Error{"create pipeline", 0,
            "no vertex shader set -- call vertexShader(path) or vertexModule(module)"};
    }
    if (!hasFragShader) {
        return Error{"create pipeline", 0,
            "no fragment shader set -- call fragmentShader(path) or fragmentModule(module)"};
    }
    if (colorFormat_ == VK_FORMAT_UNDEFINED) {
        return Error{"create pipeline", 0,
            "no color format set -- call colorFormat(swapchain) or colorFormat(VkFormat)"};
    }

    VkShaderModule vertMod = vertModule_;
    VkShaderModule fragMod = fragModule_;
    bool createdVert = false;
    bool createdFrag = false;

    // Cleanup helper: destroys internally-created modules.
    auto destroyModules = [&]() {
        if (createdVert && vertMod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, vertMod, nullptr);
        }
        if (createdFrag && fragMod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, fragMod, nullptr);
        }
    };

    if (vertMod == VK_NULL_HANDLE) {
        auto code = readSpv(vertPath_);
        if (!code.ok()) { return std::move(code).error(); }
        auto mod = createModule(code.value());
        if (!mod.ok()) { return std::move(mod).error(); }
        vertMod = mod.value();
        createdVert = true;
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

    // Local copies keep buildWithFlags() const while reflection mutates them.
    std::vector<VkDescriptorSetLayout> reflectedSetLayouts;
    std::vector<std::uint32_t> vertCode, fragCode;
    auto localPushConstantRanges   = pushConstantRanges_;
    auto localDescriptorSetLayouts = descriptorSetLayouts_;

    if (reflect_) {
        if (vertPath_.empty() || fragPath_.empty()) {
            destroyModules();
            return Error{"create pipeline", 0,
                "reflectDescriptors() requires path-based shaders "
                "(vertexShader/fragmentShader), not pre-created modules"};
        }

        auto vc = readSpv(vertPath_);
        if (!vc.ok()) { destroyModules(); return std::move(vc).error(); }
        vertCode = std::move(vc).value();

        auto fc = readSpv(fragPath_);
        if (!fc.ok()) { destroyModules(); return std::move(fc).error(); }
        fragCode = std::move(fc).value();

        auto vertRefl = reflectSpv(vertCode, VK_SHADER_STAGE_VERTEX_BIT);
        if (!vertRefl.ok()) { destroyModules(); return std::move(vertRefl).error(); }

        auto fragRefl = reflectSpv(fragCode, VK_SHADER_STAGE_FRAGMENT_BIT);
        if (!fragRefl.ok()) { destroyModules(); return std::move(fragRefl).error(); }

        auto merged = mergeReflections(vertRefl.value(), fragRefl.value());
        if (!merged.ok()) { destroyModules(); return std::move(merged).error(); }

        const auto& layout = merged.value();

        std::map<std::uint32_t, std::vector<VkDescriptorSetLayoutBinding>> bySet;
        for (const auto& rb : layout.bindings) {
            VkDescriptorSetLayoutBinding b{};
            b.binding         = rb.binding;
            b.descriptorType  = rb.type;
            b.descriptorCount = rb.count;
            b.stageFlags      = rb.stages;
            bySet[rb.set].push_back(b);
        }

        std::uint32_t maxSet = bySet.empty() ? 0 : bySet.rbegin()->first;
        for (std::uint32_t s = 0; s <= maxSet; ++s) {
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            auto it = bySet.find(s);
            if (it != bySet.end()) {
                ci.bindingCount = static_cast<std::uint32_t>(it->second.size());
                ci.pBindings    = it->second.data();
            }
            VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
            VkResult vr = vkCreateDescriptorSetLayout(device_, &ci, nullptr, &dsl);
            if (vr != VK_SUCCESS) {
                for (auto prev : reflectedSetLayouts)
                    vkDestroyDescriptorSetLayout(device_, prev, nullptr);
                destroyModules();
                return Error{"create reflected descriptor set layout",
                             static_cast<std::int32_t>(vr),
                             "vkCreateDescriptorSetLayout failed for set "
                             + std::to_string(s)};
            }
            reflectedSetLayouts.push_back(dsl);
        }

        localPushConstantRanges   = layout.pushConstants;
        localDescriptorSetLayouts = reflectedSetLayouts;
    }

    Pipeline p;
    p.device_ = device_;

    if (externalLayout_ != VK_NULL_HANDLE) {
        p.layout_ = externalLayout_;
        p.ownsLayout_ = false;
    } else {
        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount =
            static_cast<std::uint32_t>(localDescriptorSetLayouts.size());
        layoutCI.pSetLayouts =
            localDescriptorSetLayouts.empty() ? nullptr : localDescriptorSetLayouts.data();
        layoutCI.pushConstantRangeCount =
            static_cast<std::uint32_t>(localPushConstantRanges.size());
        layoutCI.pPushConstantRanges =
            localPushConstantRanges.empty() ? nullptr : localPushConstantRanges.data();

        VkResult vr = vkCreatePipelineLayout(device_, &layoutCI, nullptr, &p.layout_);
        if (vr != VK_SUCCESS) {
            for (auto dsl : reflectedSetLayouts)
                vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
            destroyModules();
            return Error{"create pipeline layout", static_cast<std::int32_t>(vr),
                "vkCreatePipelineLayout failed"};
        }
        p.ownsLayout_ = true;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

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
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology_;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = polygonMode_;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = cullMode_;
    rasterizer.frontFace   = frontFace_;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = samples_;

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

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    if (depthFormat_ != VK_FORMAT_UNDEFINED) {
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = depthCompareOp_;
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

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    // Dynamic rendering (Vulkan 1.3 core): no VkRenderPass, formats declared here instead.
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat_;
    if (depthFormat_ != VK_FORMAT_UNDEFINED) {
        renderingInfo.depthAttachmentFormat = depthFormat_;
    }

    VkPipelineCreationFeedback pipelineFeedback{};
    VkPipelineCreationFeedback stageFeedbacks[2]{};

    VkPipelineCreationFeedbackCreateInfo feedbackCI{};
    feedbackCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
    feedbackCI.pNext = &renderingInfo;
    feedbackCI.pPipelineCreationFeedback          = &pipelineFeedback;
    feedbackCI.pipelineStageCreationFeedbackCount = 2;
    feedbackCI.pPipelineStageCreationFeedbacks    = stageFeedbacks;

    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.pNext               = &feedbackCI;
    pipelineCI.flags               = flags;
    pipelineCI.stageCount          = 2;
    pipelineCI.pStages             = stages;
    pipelineCI.pVertexInputState   = &vertexInput;
    pipelineCI.pInputAssemblyState = &inputAssembly;
    pipelineCI.pViewportState      = &viewportState;
    pipelineCI.pRasterizationState = &rasterizer;
    pipelineCI.pMultisampleState   = &multisampling;
    pipelineCI.pDepthStencilState  = &depthStencil;
    pipelineCI.pColorBlendState    = &colorBlend;
    pipelineCI.pDynamicState       = &dynamicState;
    pipelineCI.layout              = p.layout_;
    pipelineCI.renderPass          = VK_NULL_HANDLE;

    VkResult vr = vkCreateGraphicsPipelines(
        device_, cache_, 1, &pipelineCI, nullptr, &p.pipeline_);

    // Always destroy internally-created shader modules.
    destroyModules();

    if (vr != VK_SUCCESS) {
        if (p.ownsLayout_ && p.layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, p.layout_, nullptr);
            p.layout_ = VK_NULL_HANDLE;
        }
        for (auto dsl : reflectedSetLayouts) {
            vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
        }
        return Error{"create pipeline", static_cast<std::int32_t>(vr),
            "vkCreateGraphicsPipelines failed"};
    }

    p.bindPoint_ = VK_PIPELINE_BIND_POINT_GRAPHICS;
    p.ownedSetLayouts_ = std::move(reflectedSetLayouts);
    for (const auto& r : localPushConstantRanges) {
        p.pcStages_ |= r.stageFlags;
        auto end = r.offset + r.size;
        if (end > p.pcSize_) p.pcSize_ = end;
    }

    PipelineStats stats;
    stats.valid      = (pipelineFeedback.flags &
                        VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
    stats.cacheHit   = (pipelineFeedback.flags &
                        VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
    stats.durationMs = static_cast<double>(pipelineFeedback.duration) / 1'000'000.0;

    VkShaderStageFlagBits stageFlags[2] = {
        VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT
    };
    for (int i = 0; i < 2; ++i) {
        StageFeedback sf;
        sf.stage      = stageFlags[i];
        sf.valid      = (stageFeedbacks[i].flags &
                         VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
        sf.cacheHit   = (stageFeedbacks[i].flags &
                         VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
        sf.durationMs = static_cast<double>(stageFeedbacks[i].duration) / 1'000'000.0;
        stats.stages.push_back(sf);
    }
    p.stats_ = std::move(stats);

    return p;
}

} // namespace vksdl
