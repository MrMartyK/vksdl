#pragma once

#include <vulkan/vulkan.h>

namespace vksdl::detail {

// Dynamically loaded RT extension functions.
// Loaded once per device via loadRtFunctions(). All pointers are null until loaded.
// Same pattern as debug.cpp with PFN_vkSetDebugUtilsObjectNameEXT.

struct RtFunctions {
    PFN_vkCreateAccelerationStructureKHR              createAs               = nullptr;
    PFN_vkDestroyAccelerationStructureKHR             destroyAs              = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR            cmdBuildAs             = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR       getBuildSizes          = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR    getAsDeviceAddress     = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR cmdWriteAsProperties   = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR             cmdCopyAs              = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR                createRtPipelines      = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR          getShaderGroupHandles  = nullptr;
    PFN_vkCmdTraceRaysKHR                             cmdTraceRays           = nullptr;
};

// Load all RT extension function pointers for the given device.
// Returns a struct with loaded pointers. Any function not available is null.
RtFunctions loadRtFunctions(VkDevice device);

} // namespace vksdl::detail
