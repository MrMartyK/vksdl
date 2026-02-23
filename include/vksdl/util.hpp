#pragma once

#include <vulkan/vulkan.h>

#include <filesystem>

namespace vksdl {

[[nodiscard]] inline VkDeviceSize alignUp(VkDeviceSize size, VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Returns the directory containing the running executable.
// Uses SDL_GetBasePath() internally.
[[nodiscard]] std::filesystem::path exeDir();

// Returns exeDir() / relative.
[[nodiscard]] std::filesystem::path exeRelativePath(
    const std::filesystem::path& relative);

} // namespace vksdl
