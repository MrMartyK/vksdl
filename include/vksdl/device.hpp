#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vksdl {

class Instance;
class Surface;

enum class GpuPrefer {
    Discrete,
    Integrated,
    Any,
};

struct QueueFamilies {
    std::uint32_t graphics = UINT32_MAX;
    std::uint32_t present  = UINT32_MAX;
    std::uint32_t transfer = UINT32_MAX; // dedicated transfer, UINT32_MAX = none
    std::uint32_t compute  = UINT32_MAX; // dedicated compute,  UINT32_MAX = none

    [[nodiscard]] bool valid() const {
        return graphics != UINT32_MAX && present != UINT32_MAX;
    }

    [[nodiscard]] bool shared() const {
        return graphics == present;
    }

    [[nodiscard]] bool hasDedicatedTransfer() const {
        return transfer != UINT32_MAX && transfer != graphics;
    }

    [[nodiscard]] bool hasDedicatedCompute() const {
        return compute != UINT32_MAX && compute != graphics;
    }
};

// Thread safety: immutable after construction. VkQueue handles returned by
// accessors follow Vulkan queue externally-synchronized rules.
class Device {
public:
    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] VkDevice         native()           const { return device_; }
    [[nodiscard]] VkDevice         vkDevice()         const { return native(); }
    [[nodiscard]] VkPhysicalDevice vkPhysicalDevice() const { return physicalDevice_; }
    [[nodiscard]] VkQueue          graphicsQueue()     const { return graphicsQueue_; }
    [[nodiscard]] VkQueue          presentQueue()      const { return presentQueue_; }
    [[nodiscard]] VkQueue          transferQueue()     const { return transferQueue_; }
    [[nodiscard]] VkQueue          computeQueue()      const { return computeQueue_; }
    [[nodiscard]] QueueFamilies    queueFamilies()     const { return families_; }
    [[nodiscard]] bool             hasDedicatedTransfer() const { return families_.hasDedicatedTransfer(); }
    [[nodiscard]] bool             hasAsyncCompute()      const { return families_.hasDedicatedCompute(); }
    [[nodiscard]] bool             hasDedicatedCompute()  const { return families_.hasDedicatedCompute(); }
    [[nodiscard]] VkDeviceSize           minUniformBufferOffsetAlignment() const { return minUboAlignment_; }
    [[nodiscard]] VkSampleCountFlagBits  maxMsaaSamples()  const { return maxMsaaSamples_; }
    [[nodiscard]] float                  timestampPeriod() const { return timestampPeriod_; }
    [[nodiscard]] const char*             gpuName()         const { return gpuName_.c_str(); }

    // RT properties (0 when RT not requested)
    [[nodiscard]] std::uint32_t shaderGroupHandleSize()      const { return rtHandleSize_; }
    [[nodiscard]] std::uint32_t shaderGroupBaseAlignment()   const { return rtBaseAlignment_; }
    [[nodiscard]] std::uint32_t shaderGroupHandleAlignment() const { return rtHandleAlignment_; }
    [[nodiscard]] std::uint32_t maxRayRecursionDepth()       const { return rtMaxRecursion_; }
    [[nodiscard]] VkDeviceSize  minAccelerationStructureScratchOffsetAlignment() const { return rtScratchAlignment_; }

    [[nodiscard]] PFN_vkCmdTraceRaysKHR traceRaysFn() const { return pfnTraceRays_; }

    // VK_EXT_device_fault support (opportunistic -- always safe to call)
    [[nodiscard]] bool hasDeviceFault() const { return hasDeviceFault_; }

    // VK_EXT_memory_budget support (opportunistic).
    // When true, Allocator::queryBudget() returns accurate OS-reported values.
    [[nodiscard]] bool hasMemoryBudget() const { return hasMemoryBudget_; }

    // VK_EXT_memory_priority support (opportunistic).
    // When true, BufferBuilder/ImageBuilder::memoryPriority(p) influences
    // which allocations are evicted first under memory pressure.
    [[nodiscard]] bool hasMemoryPriority() const { return hasMemoryPriority_; }

    // VK_KHR_unified_image_layouts support (extremely new extension).
    // When true, images can stay in VK_IMAGE_LAYOUT_GENERAL without explicit
    // transitions. Barriers remain conservative by default -- this is detect-
    // and-expose only. Sync validation becomes mandatory when using unified
    // layouts since validation loses layout-mismatch detection ability.
    [[nodiscard]] bool hasUnifiedImageLayouts() const { return hasUnifiedLayouts_; }

    // VK_EXT_graphics_pipeline_library support (opportunistic detection).
    // hasGPL() -- extension present and features enabled.
    // hasGplFastLinking() -- driver can link GPL parts without optimization cost.
    // hasGplIndependentInterpolation() -- per-component interpolation qualifiers
    //   are safe under GPL. When false (AMDVLK), GPL may produce incorrect
    //   results for shaders with mixed interpolation qualifiers.
    [[nodiscard]] bool hasGPL()                        const { return hasGPL_; }
    [[nodiscard]] bool hasGplFastLinking()              const { return hasGplFastLinking_; }
    [[nodiscard]] bool hasGplIndependentInterpolation() const { return hasGplIndepInterp_; }

    // pipelineCreationCacheControl (Vulkan 1.3 core feature).
    // When true, FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT is usable for zero-cost
    // cache probes. Some 1.3 drivers report this as unsupported.
    [[nodiscard]] bool hasPipelineCreationCacheControl() const { return hasPCCC_; }

    // VK_KHR_push_descriptor support (opportunistic detection).
    // When true, descriptors can be pushed directly into command buffers
    // via vkCmdPushDescriptorSetKHR, avoiding descriptor pool allocation.
    [[nodiscard]] bool hasPushDescriptors() const { return hasPushDescriptors_; }

    // Bindless descriptor support (Vulkan 1.2 descriptor indexing features).
    // When true, BindlessTable can be used for large partially-bound arrays
    // with UPDATE_AFTER_BIND. Requires descriptorBindingPartiallyBound,
    // runtimeDescriptorArray, and at least one updateAfterBind feature.
    [[nodiscard]] bool hasBindless() const { return hasBindless_; }

    // VK_NV_ray_tracing_invocation_reorder (SER) support.
    // When true, the driver reorders RT shader invocations for better
    // coherence. Zero behavioral impact -- purely a performance optimization.
    [[nodiscard]] bool hasInvocationReorder() const { return hasInvocationReorder_; }

    // VK_KHR_pipeline_binary support (opportunistic detection).
    // When true, pipeline binaries can be captured and reloaded to skip
    // shader compilation on subsequent runs.
    [[nodiscard]] bool hasPipelineBinary() const { return hasPipelineBinary_; }

    // VK_EXT_mesh_shader support.
    // When true, MeshPipelineBuilder is usable and drawMeshTasksFn() is valid.
    [[nodiscard]] bool hasMeshShaders() const { return hasMeshShaders_; }
    [[nodiscard]] PFN_vkCmdDrawMeshTasksEXT drawMeshTasksFn() const { return pfnDrawMeshTasks_; }

    [[nodiscard]] std::string queryDeviceFault() const;

    void waitIdle() const;

private:
    friend class DeviceBuilder;
    Device() = default;

    VkDevice         device_         = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    VkQueue          transferQueue_  = VK_NULL_HANDLE;
    VkQueue          computeQueue_   = VK_NULL_HANDLE;
    QueueFamilies         families_;
    VkDeviceSize          minUboAlignment_  = 256;
    VkSampleCountFlagBits maxMsaaSamples_   = VK_SAMPLE_COUNT_1_BIT;
    float                 timestampPeriod_  = 0.0f;
    std::string           gpuName_;

    // Device fault
    bool hasDeviceFault_ = false;
    // Memory budget and priority extensions
    bool hasMemoryBudget_   = false;
    bool hasMemoryPriority_ = false;
    // Unified image layouts
    bool hasUnifiedLayouts_ = false;
    // Graphics pipeline library
    bool hasGPL_             = false;
    bool hasGplFastLinking_  = false;
    bool hasGplIndepInterp_  = false;
    // Pipeline creation cache control (Vulkan 1.3 core feature)
    bool hasPCCC_ = false;
    // Push descriptors
    bool hasPushDescriptors_ = false;
    // Bindless descriptors (Vulkan 1.2 descriptor indexing)
    bool hasBindless_ = false;
    // SER (shader execution reorder)
    bool hasInvocationReorder_ = false;
    // Pipeline binary (VK_KHR_pipeline_binary)
    bool hasPipelineBinary_ = false;
    // Mesh shaders (VK_EXT_mesh_shader)
    bool hasMeshShaders_ = false;

    // RT properties
    PFN_vkCmdTraceRaysKHR pfnTraceRays_    = nullptr;
    // Mesh shader function pointer (null when not loaded)
    PFN_vkCmdDrawMeshTasksEXT pfnDrawMeshTasks_ = nullptr;
    std::uint32_t rtHandleSize_      = 0;
    std::uint32_t rtBaseAlignment_   = 0;
    std::uint32_t rtHandleAlignment_ = 0;
    std::uint32_t rtMaxRecursion_    = 0;
    VkDeviceSize  rtScratchAlignment_ = 0;
};

class DeviceBuilder {
public:
    // Takes vksdl wrapper objects -- no raw handles needed.
    DeviceBuilder(const Instance& instance, const Surface& surface);

    DeviceBuilder& needSwapchain();
    DeviceBuilder& needDynamicRendering();
    DeviceBuilder& needSync2();
    DeviceBuilder& graphicsDefaults();
    DeviceBuilder& needRayTracingPipeline();
    DeviceBuilder& needRayQuery();
    DeviceBuilder& needGPL();
    DeviceBuilder& needMeshShaders(); // requires VK_EXT_mesh_shader
    DeviceBuilder& needAsyncCompute(); // preference, not requirement -- falls back to graphics
    DeviceBuilder& preferDiscreteGpu();
    DeviceBuilder& preferIntegratedGpu();

    DeviceBuilder& requireExtension(const char* name);
    DeviceBuilder& requireFeatures(
        std::function<void(VkPhysicalDeviceVulkan13Features&)> configureFn);
    DeviceBuilder& requireFeatures12(
        std::function<void(VkPhysicalDeviceVulkan12Features&)> configureFn);
    DeviceBuilder& requireCoreFeatures(
        std::function<void(VkPhysicalDeviceFeatures&)> configureFn);

    // Chain an arbitrary extension feature struct into VkDeviceCreateInfo::pNext.
    // The struct must have sType set. Caller keeps the struct alive until build().
    DeviceBuilder& chainFeatures(void* featureStruct);

    DeviceBuilder& preferGpu(GpuPrefer pref);

    [[nodiscard]] Result<Device> build();

private:
    struct FeatureRequest13 {
        std::function<void(VkPhysicalDeviceVulkan13Features&)> configure;
    };
    struct FeatureRequest12 {
        std::function<void(VkPhysicalDeviceVulkan12Features&)> configure;
    };
    struct CoreFeatureRequest {
        std::function<void(VkPhysicalDeviceFeatures&)> configure;
    };

    [[nodiscard]] QueueFamilies findQueueFamilies(VkPhysicalDevice gpu) const;
    [[nodiscard]] bool          supportsExtensions(VkPhysicalDevice gpu) const;
    [[nodiscard]] int           scoreDevice(VkPhysicalDevice gpu) const;

    VkInstance                      instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR                    surface_  = VK_NULL_HANDLE;
    GpuPrefer                       gpuPref_  = GpuPrefer::Discrete;
    std::vector<const char*>        extensions_;
    std::vector<FeatureRequest13>   featureRequests_;
    std::vector<FeatureRequest12>   featureRequests12_;
    std::vector<CoreFeatureRequest> coreFeatureRequests_;
    std::vector<void*>              chainedFeatures_;
    bool needRayTracingPipeline_    = false;
    bool needRayQuery_              = false;
    bool needAccelerationStructure_ = false;
    bool needGPL_                   = false;
    bool needMeshShaders_           = false;
    bool needAsyncCompute_          = false;
};

} // namespace vksdl
