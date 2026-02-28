#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vksdl {

// A single descriptor binding discovered via shader reflection.
struct ReflectedBinding {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    std::uint32_t count = 1;
    VkShaderStageFlags stages = 0;
    std::string name; // GLSL binding name (e.g. "shadowDepth"), from SPIR-V metadata.
};

// Complete reflected layout for one or more shader stages.
struct ReflectedLayout {
    std::vector<ReflectedBinding> bindings;
    std::vector<VkPushConstantRange> pushConstants;
};

// Reflect a single SPIR-V module. Returns bindings and push constants
// for the given shader stage.
[[nodiscard]] Result<ReflectedLayout> reflectSpv(const std::vector<std::uint32_t>& code,
                                                 VkShaderStageFlags stage);

// Reflect a SPIR-V file from disk.
[[nodiscard]] Result<ReflectedLayout> reflectSpvFile(const std::filesystem::path& path,
                                                     VkShaderStageFlags stage);

// Merge two reflected layouts. Same set+binding+type: stage flags are OR'd.
// Same set+binding but different type: returns an Error.
[[nodiscard]] Result<ReflectedLayout> mergeReflections(const ReflectedLayout& a,
                                                       const ReflectedLayout& b);

} // namespace vksdl
