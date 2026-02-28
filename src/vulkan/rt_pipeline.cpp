#include <vksdl/device.hpp>
#include <vksdl/pipeline_cache.hpp>
#include <vksdl/rt_pipeline.hpp>

#include "rt_functions.hpp"

#include <cstdint>
#include <vector>

namespace vksdl {

RayTracingPipelineBuilder::RayTracingPipelineBuilder(const Device& device)
    : device_(device.vkDevice()) {}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::rayGenShader(const std::filesystem::path& spvPath) {
    stages_.push_back({StageType::RayGen, spvPath, VK_NULL_HANDLE});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::missShader(const std::filesystem::path& spvPath) {
    stages_.push_back({StageType::Miss, spvPath, VK_NULL_HANDLE});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::closestHitShader(const std::filesystem::path& spvPath) {
    stages_.push_back({StageType::ClosestHit, spvPath, VK_NULL_HANDLE});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::anyHitShader(const std::filesystem::path& spvPath) {
    stages_.push_back({StageType::AnyHit, spvPath, VK_NULL_HANDLE});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::intersectionShader(const std::filesystem::path& spvPath) {
    stages_.push_back({StageType::Intersection, spvPath, VK_NULL_HANDLE});
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::rayGenModule(VkShaderModule module) {
    stages_.push_back({StageType::RayGen, {}, module});
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::missModule(VkShaderModule module) {
    stages_.push_back({StageType::Miss, {}, module});
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::closestHitModule(VkShaderModule module) {
    stages_.push_back({StageType::ClosestHit, {}, module});
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::anyHitModule(VkShaderModule module) {
    stages_.push_back({StageType::AnyHit, {}, module});
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::intersectionModule(VkShaderModule module) {
    stages_.push_back({StageType::Intersection, {}, module});
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::addTrianglesHitGroup(std::uint32_t closestHitIndex,
                                                std::uint32_t anyHitIndex) {
    HitGroupEntry hg{};
    hg.closestHitIndex = closestHitIndex;
    hg.anyHitIndex = anyHitIndex;
    hitGroups_.push_back(hg);
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::addProceduralHitGroup(
    std::uint32_t intersectionIndex, std::uint32_t closestHitIndex, std::uint32_t anyHitIndex) {
    HitGroupEntry hg{};
    hg.procedural = true;
    hg.intersectionIndex = intersectionIndex;
    hg.closestHitIndex = closestHitIndex;
    hg.anyHitIndex = anyHitIndex;
    hitGroups_.push_back(hg);
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::maxRecursionDepth(std::uint32_t depth) {
    maxRecursion_ = depth;
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::specialize(const VkSpecializationInfo& info) {
    externalSpecInfo_ = info;
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::cache(const PipelineCache& c) {
    cache_ = c.vkPipelineCache();
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::cache(VkPipelineCache c) {
    cache_ = c;
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::pushConstantRange(VkPushConstantRange range) {
    pushConstantRanges_.push_back(range);
    return *this;
}

RayTracingPipelineBuilder&
RayTracingPipelineBuilder::descriptorSetLayout(VkDescriptorSetLayout layout) {
    descriptorSetLayouts_.push_back(layout);
    return *this;
}

RayTracingPipelineBuilder& RayTracingPipelineBuilder::pipelineLayout(VkPipelineLayout layout) {
    externalLayout_ = layout;
    return *this;
}

Result<VkShaderModule>
RayTracingPipelineBuilder::createModule(const std::vector<std::uint32_t>& code) const {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size() * sizeof(std::uint32_t);
    ci.pCode = code.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult vr = vkCreateShaderModule(device_, &ci, nullptr, &module);
    if (vr != VK_SUCCESS) {
        return Error{"create RT shader module", static_cast<std::int32_t>(vr),
                     "vkCreateShaderModule failed"};
    }
    return module;
}

Result<Pipeline> RayTracingPipelineBuilder::build() {
    // Validate: need at least one raygen shader.
    bool hasRaygen = false;
    for (const auto& s : stages_) {
        if (s.type == StageType::RayGen) {
            hasRaygen = true;
            break;
        }
    }
    if (!hasRaygen) {
        return Error{"create RT pipeline", 0,
                     "no raygen shader -- call rayGenShader() or rayGenModule()"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.createRtPipelines) {
        return Error{"create RT pipeline", 0,
                     "vkCreateRayTracingPipelinesKHR not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    std::vector<VkShaderModule> createdModules;
    auto destroyCreatedModules = [&]() {
        for (auto m : createdModules) {
            vkDestroyShaderModule(device_, m, nullptr);
        }
    };

    // Track global stage indices per type so hit group construction can map
    // per-category indices (e.g., closestHit[0]) to global stage indices.
    std::vector<VkPipelineShaderStageCreateInfo> vkStages;
    std::vector<std::uint32_t> raygenGlobalIndices;
    std::vector<std::uint32_t> missGlobalIndices;
    std::vector<std::uint32_t> closestHitGlobalIndices;
    std::vector<std::uint32_t> anyHitGlobalIndices;
    std::vector<std::uint32_t> intersectionGlobalIndices;

    // Build specialization info (shared across all RT stages).
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

    for (const auto& entry : stages_) {
        VkShaderModule mod = entry.module;
        if (mod == VK_NULL_HANDLE) {
            auto code = readSpv(entry.path);
            if (!code.ok()) {
                destroyCreatedModules();
                return std::move(code).error();
            }
            auto modResult = createModule(code.value());
            if (!modResult.ok()) {
                destroyCreatedModules();
                return std::move(modResult).error();
            }
            mod = modResult.value();
            createdModules.push_back(mod);
        }

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.module = mod;
        stage.pName = "main";
        stage.pSpecializationInfo = pSpecInfo;

        auto globalIdx = static_cast<std::uint32_t>(vkStages.size());

        switch (entry.type) {
        case StageType::RayGen:
            stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            raygenGlobalIndices.push_back(globalIdx);
            break;
        case StageType::Miss:
            stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            missGlobalIndices.push_back(globalIdx);
            break;
        case StageType::ClosestHit:
            stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            closestHitGlobalIndices.push_back(globalIdx);
            break;
        case StageType::AnyHit:
            stage.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            anyHitGlobalIndices.push_back(globalIdx);
            break;
        case StageType::Intersection:
            stage.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
            intersectionGlobalIndices.push_back(globalIdx);
            break;
        }

        vkStages.push_back(stage);
    }

    // Shader group order: raygen, miss, hit (matches SBT region layout).
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

    for (auto idx : raygenGlobalIndices) {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader = idx;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(g);
    }

    for (auto idx : missGlobalIndices) {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader = idx;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(g);
    }

    if (hitGroups_.empty()) {
        // Auto-grouping: one TRIANGLES_HIT_GROUP per closest-hit shader.
        for (auto idx : closestHitGlobalIndices) {
            VkRayTracingShaderGroupCreateInfoKHR g{};
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            g.generalShader = VK_SHADER_UNUSED_KHR;
            g.closestHitShader = idx;
            g.anyHitShader = VK_SHADER_UNUSED_KHR;
            g.intersectionShader = VK_SHADER_UNUSED_KHR;
            groups.push_back(g);
        }
    } else {
        // Explicit hit groups: per-category indices (e.g., closestHit[2])
        // are converted to global stage indices here.
        for (const auto& hg : hitGroups_) {
            VkRayTracingShaderGroupCreateInfoKHR g{};
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader = VK_SHADER_UNUSED_KHR;

            if (hg.procedural) {
                g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
                g.intersectionShader = hg.intersectionIndex < intersectionGlobalIndices.size()
                                           ? intersectionGlobalIndices[hg.intersectionIndex]
                                           : VK_SHADER_UNUSED_KHR;
            } else {
                g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                g.intersectionShader = VK_SHADER_UNUSED_KHR;
            }

            g.closestHitShader = hg.closestHitIndex < closestHitGlobalIndices.size()
                                     ? closestHitGlobalIndices[hg.closestHitIndex]
                                     : VK_SHADER_UNUSED_KHR;

            g.anyHitShader = hg.anyHitIndex < anyHitGlobalIndices.size()
                                 ? anyHitGlobalIndices[hg.anyHitIndex]
                                 : VK_SHADER_UNUSED_KHR;

            groups.push_back(g);
        }
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
            destroyCreatedModules();
            return Error{"create RT pipeline layout", static_cast<std::int32_t>(vr),
                         "vkCreatePipelineLayout failed"};
        }
        p.ownsLayout_ = true;
    }

    VkPipelineCreationFeedback pipelineFeedback{};
    std::vector<VkPipelineCreationFeedback> rtStageFeedbacks(vkStages.size());

    VkPipelineCreationFeedbackCreateInfo feedbackCI{};
    feedbackCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
    feedbackCI.pPipelineCreationFeedback = &pipelineFeedback;
    feedbackCI.pipelineStageCreationFeedbackCount = static_cast<std::uint32_t>(vkStages.size());
    feedbackCI.pPipelineStageCreationFeedbacks = rtStageFeedbacks.data();

    VkRayTracingPipelineCreateInfoKHR pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineCI.pNext = &feedbackCI;
    pipelineCI.stageCount = static_cast<std::uint32_t>(vkStages.size());
    pipelineCI.pStages = vkStages.data();
    pipelineCI.groupCount = static_cast<std::uint32_t>(groups.size());
    pipelineCI.pGroups = groups.data();
    pipelineCI.maxPipelineRayRecursionDepth = maxRecursion_;
    pipelineCI.layout = p.layout_;

    VkResult vr = fn.createRtPipelines(device_, VK_NULL_HANDLE, cache_, 1, &pipelineCI, nullptr,
                                       &p.pipeline_);

    destroyCreatedModules();

    if (vr != VK_SUCCESS) {
        if (p.ownsLayout_ && p.layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, p.layout_, nullptr);
            p.layout_ = VK_NULL_HANDLE;
        }
        return Error{"create RT pipeline", static_cast<std::int32_t>(vr),
                     "vkCreateRayTracingPipelinesKHR failed"};
    }

    p.bindPoint_ = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
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

    for (std::size_t i = 0; i < vkStages.size(); ++i) {
        StageFeedback sf;
        sf.stage = static_cast<VkShaderStageFlagBits>(vkStages[i].stage);
        sf.valid = (rtStageFeedbacks[i].flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
        sf.cacheHit = (rtStageFeedbacks[i].flags &
                       VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
        sf.durationMs = static_cast<double>(rtStageFeedbacks[i].duration) / 1'000'000.0;
        stats.stages.push_back(sf);
    }
    p.stats_ = std::move(stats);

    return p;
}

} // namespace vksdl
