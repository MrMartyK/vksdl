#include <vksdl/compute_pipeline.hpp>
#include <vksdl/device.hpp>
#include <vksdl/pipeline_cache.hpp>
#include <vksdl/shader_reflect.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vksdl {

ComputePipelineBuilder::ComputePipelineBuilder(const Device& device)
    : device_(device.vkDevice()) {}

ComputePipelineBuilder& ComputePipelineBuilder::shader(
    const std::filesystem::path& spvPath) {
    shaderPath_ = spvPath;
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::shaderModule(VkShaderModule module) {
    shaderModule_ = module;
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::specialize(
    const VkSpecializationInfo& info) {
    externalSpecInfo_ = info;
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::cache(const PipelineCache& c) {
    cache_ = c.vkPipelineCache();
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::pushConstantRange(
    VkPushConstantRange range) {
    pushConstantRanges_.push_back(range);
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::descriptorSetLayout(
    VkDescriptorSetLayout layout) {
    descriptorSetLayouts_.push_back(layout);
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::pipelineLayout(
    VkPipelineLayout layout) {
    externalLayout_ = layout;
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::reflectDescriptors() {
    reflect_ = true;
    return *this;
}

Result<VkShaderModule> ComputePipelineBuilder::createModule(
    const std::vector<std::uint32_t>& code) const {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size() * sizeof(std::uint32_t);
    ci.pCode    = code.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(device_, &ci, nullptr, &module);
    if (vr != VK_SUCCESS) {
        return Error{"create compute shader module",
                     static_cast<std::int32_t>(vr),
                     "vkCreateShaderModule failed"};
    }
    return module;
}

Result<Pipeline> ComputePipelineBuilder::build() {
    bool hasShader = !shaderPath_.empty() || shaderModule_ != VK_NULL_HANDLE;
    if (!hasShader) {
        return Error{"create compute pipeline", 0,
            "no compute shader set -- call shader(path) or shaderModule(module)"};
    }

    VkShaderModule compMod = shaderModule_;
    bool createdModule = false;

    auto destroyModule = [&]() {
        if (createdModule && compMod != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, compMod, nullptr);
        }
    };

    if (compMod == VK_NULL_HANDLE) {
        auto code = readSpv(shaderPath_);
        if (!code.ok()) { return std::move(code).error(); }
        auto mod = createModule(code.value());
        if (!mod.ok()) { return std::move(mod).error(); }
        compMod = mod.value();
        createdModule = true;
    }

    std::vector<VkDescriptorSetLayout> reflectedSetLayouts;
    std::optional<ReflectedLayout> localReflectedLayout;
    std::vector<VkPushConstantRange> localPushConstantRanges;
    std::vector<VkDescriptorSetLayout> localDescriptorSetLayouts;

    if (reflect_) {
        if (shaderPath_.empty()) {
            destroyModule();
            return Error{"create compute pipeline", 0,
                "reflectDescriptors() requires path-based shader, "
                "not a pre-created module"};
        }

        auto code = readSpv(shaderPath_);
        if (!code.ok()) { destroyModule(); return std::move(code).error(); }

        auto refl = reflectSpv(code.value(), VK_SHADER_STAGE_COMPUTE_BIT);
        if (!refl.ok()) { destroyModule(); return std::move(refl).error(); }

        const auto& layout = refl.value();
        localReflectedLayout = layout;

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
                destroyModule();
                return Error{"create reflected descriptor set layout",
                             static_cast<std::int32_t>(vr),
                             "vkCreateDescriptorSetLayout failed for set "
                             + std::to_string(s)};
            }
            reflectedSetLayouts.push_back(dsl);
        }

        localPushConstantRanges = layout.pushConstants;
        localDescriptorSetLayouts = reflectedSetLayouts;
    }

    // Use locals so build() doesn't mutate builder state.
    const auto& activePCRanges = localPushConstantRanges.empty()
        ? pushConstantRanges_ : localPushConstantRanges;
    const auto& activeDSLayouts = localDescriptorSetLayouts.empty()
        ? descriptorSetLayouts_ : localDescriptorSetLayouts;

    Pipeline p;
    p.device_ = device_;

    if (externalLayout_ != VK_NULL_HANDLE) {
        p.layout_ = externalLayout_;
        p.ownsLayout_ = false;
    } else {
        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount =
            static_cast<std::uint32_t>(activeDSLayouts.size());
        layoutCI.pSetLayouts =
            activeDSLayouts.empty() ? nullptr : activeDSLayouts.data();
        layoutCI.pushConstantRangeCount =
            static_cast<std::uint32_t>(activePCRanges.size());
        layoutCI.pPushConstantRanges =
            activePCRanges.empty() ? nullptr : activePCRanges.data();

        VkResult vr = vkCreatePipelineLayout(device_, &layoutCI, nullptr, &p.layout_);
        if (vr != VK_SUCCESS) {
            for (auto dsl : reflectedSetLayouts)
                vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
            destroyModule();
            return Error{"create compute pipeline layout",
                         static_cast<std::int32_t>(vr),
                         "vkCreatePipelineLayout failed"};
        }
        p.ownsLayout_ = true;
    }

    VkSpecializationInfo builtSpecInfo{};
    const VkSpecializationInfo* pSpecInfo = nullptr;
    if (externalSpecInfo_) {
        pSpecInfo = &*externalSpecInfo_;
    } else if (!specEntries_.empty()) {
        builtSpecInfo.mapEntryCount = static_cast<std::uint32_t>(specEntries_.size());
        builtSpecInfo.pMapEntries   = specEntries_.data();
        builtSpecInfo.dataSize      = specData_.size();
        builtSpecInfo.pData         = specData_.data();
        pSpecInfo = &builtSpecInfo;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module              = compMod;
    stage.pName               = "main";
    stage.pSpecializationInfo = pSpecInfo;

    VkPipelineCreationFeedback pipelineFeedback{};
    VkPipelineCreationFeedback stageFeedback{};

    VkPipelineCreationFeedbackCreateInfo feedbackCI{};
    feedbackCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
    feedbackCI.pPipelineCreationFeedback          = &pipelineFeedback;
    feedbackCI.pipelineStageCreationFeedbackCount = 1;
    feedbackCI.pPipelineStageCreationFeedbacks    = &stageFeedback;

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.pNext  = &feedbackCI;
    pipelineCI.stage  = stage;
    pipelineCI.layout = p.layout_;

    VkResult vr = vkCreateComputePipelines(
        device_, cache_, 1, &pipelineCI, nullptr, &p.pipeline_);

    destroyModule();

    if (vr != VK_SUCCESS) {
        if (p.ownsLayout_ && p.layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, p.layout_, nullptr);
            p.layout_ = VK_NULL_HANDLE;
        }
        for (auto dsl : reflectedSetLayouts) {
            vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
        }
        return Error{"create compute pipeline", static_cast<std::int32_t>(vr),
            "vkCreateComputePipelines failed"};
    }

    p.bindPoint_ = VK_PIPELINE_BIND_POINT_COMPUTE;
    p.ownedSetLayouts_ = std::move(reflectedSetLayouts);
    p.reflectedLayout_ = std::move(localReflectedLayout);
    for (const auto& r : pushConstantRanges_) {
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

    StageFeedback sf;
    sf.stage      = VK_SHADER_STAGE_COMPUTE_BIT;
    sf.valid      = (stageFeedback.flags &
                     VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
    sf.cacheHit   = (stageFeedback.flags &
                     VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
    sf.durationMs = static_cast<double>(stageFeedback.duration) / 1'000'000.0;
    stats.stages.push_back(sf);
    p.stats_ = std::move(stats);

    return p;
}

} // namespace vksdl
