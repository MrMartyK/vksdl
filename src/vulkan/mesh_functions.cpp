#include "mesh_functions.hpp"

namespace vksdl::detail {

MeshFunctions loadMeshFunctions(VkDevice device) {
    MeshFunctions fn;
    fn.cmdDrawMeshTasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT"));
    return fn;
}

} // namespace vksdl::detail
