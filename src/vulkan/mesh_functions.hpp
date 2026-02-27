#pragma once
#include <vulkan/vulkan.h>

namespace vksdl::detail {

// Dynamically loaded VK_EXT_mesh_shader extension functions.
// Loaded once per device via loadMeshFunctions().
struct MeshFunctions {
    PFN_vkCmdDrawMeshTasksEXT cmdDrawMeshTasks = nullptr;
};

MeshFunctions loadMeshFunctions(VkDevice device);

} // namespace vksdl::detail
