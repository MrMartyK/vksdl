// SPIRV-Reflect: compiled as a single translation unit, same pattern as VMA.
// Suppress warnings from third-party code.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Woverflow"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif

#include <spirv_reflect.h>
#include <spirv_reflect.c>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <vksdl/shader_reflect.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace vksdl {

static VkDescriptorType toVkType(SpvReflectDescriptorType t) {
    switch (t) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    default:
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

Result<ReflectedLayout> reflectSpv(const std::vector<std::uint32_t>& code,
                                    VkShaderStageFlags stage) {
    SpvReflectShaderModule module{};
    SpvReflectResult result = spvReflectCreateShaderModule(
        code.size() * sizeof(std::uint32_t),
        code.data(), &module);

    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        return Error{"reflect SPIR-V", static_cast<std::int32_t>(result),
                     "spvReflectCreateShaderModule failed"};
    }

    ReflectedLayout layout;

    std::uint32_t bindingCount = 0;
    result = spvReflectEnumerateDescriptorBindings(&module, &bindingCount, nullptr);
    if (result == SPV_REFLECT_RESULT_SUCCESS && bindingCount > 0) {
        std::vector<SpvReflectDescriptorBinding*> bindings(bindingCount);
        spvReflectEnumerateDescriptorBindings(&module, &bindingCount, bindings.data());

        for (auto* b : bindings) {
            ReflectedBinding rb;
            rb.set     = b->set;
            rb.binding = b->binding;
            rb.type    = toVkType(b->descriptor_type);
            rb.count   = b->count;
            rb.stages  = stage;
            if (b->name) rb.name = b->name;
            layout.bindings.push_back(rb);
        }
    }

    std::uint32_t pcCount = 0;
    result = spvReflectEnumeratePushConstantBlocks(&module, &pcCount, nullptr);
    if (result == SPV_REFLECT_RESULT_SUCCESS && pcCount > 0) {
        std::vector<SpvReflectBlockVariable*> blocks(pcCount);
        spvReflectEnumeratePushConstantBlocks(&module, &pcCount, blocks.data());

        for (auto* block : blocks) {
            VkPushConstantRange range{};
            range.stageFlags = stage;
            range.offset     = block->offset;
            range.size       = block->size;
            layout.pushConstants.push_back(range);
        }
    }

    spvReflectDestroyShaderModule(&module);

    return layout;
}

Result<ReflectedLayout> reflectSpvFile(const std::filesystem::path& path,
                                        VkShaderStageFlags stage) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return Error{"reflect SPIR-V file", 0,
                     "failed to open: " + path.string()};
    }

    auto size = file.tellg();
    file.seekg(0);

    if (size % 4 != 0) {
        return Error{"reflect SPIR-V file", 0,
                     "file size not aligned to 4 bytes: " + path.string()};
    }

    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);

    return reflectSpv(code, stage);
}

Result<ReflectedLayout> mergeReflections(const ReflectedLayout& a,
                                          const ReflectedLayout& b) {
    ReflectedLayout merged;
    merged.bindings = a.bindings;

    for (const auto& bb : b.bindings) {
        bool found = false;
        for (auto& mb : merged.bindings) {
            if (mb.set == bb.set && mb.binding == bb.binding) {
                if (mb.type != bb.type) {
                    return Error{"merge shader reflections", 0,
                                 "binding " + std::to_string(bb.binding)
                                 + " in set " + std::to_string(bb.set)
                                 + " has conflicting types between stages"};
                }
                mb.stages |= bb.stages;
                if (mb.name.empty() && !bb.name.empty())
                    mb.name = bb.name;
                found = true;
                break;
            }
        }
        if (!found) {
            merged.bindings.push_back(bb);
        }
    }

    // Overlapping ranges: OR stage flags. Non-overlapping: append as-is.
    merged.pushConstants = a.pushConstants;
    for (const auto& bpc : b.pushConstants) {
        bool found = false;
        for (auto& mpc : merged.pushConstants) {
            if (mpc.offset == bpc.offset && mpc.size == bpc.size) {
                mpc.stageFlags |= bpc.stageFlags;
                found = true;
                break;
            }
        }
        if (!found) {
            merged.pushConstants.push_back(bpc);
        }
    }

    return merged;
}

} // namespace vksdl
