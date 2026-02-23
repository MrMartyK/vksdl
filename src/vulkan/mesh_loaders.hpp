#pragma once

// Internal header -- not part of the public API.
// Declares per-format loader functions called by loadModel() in mesh.cpp.

#include <vksdl/mesh.hpp>
#include <vksdl/result.hpp>

#include <filesystem>
#include <vector>

namespace vksdl::detail {

[[nodiscard]] Result<std::vector<MeshData>> loadGltf(const std::filesystem::path& path);
[[nodiscard]] Result<std::vector<MeshData>> loadObj(const std::filesystem::path& path);

} // namespace vksdl::detail
