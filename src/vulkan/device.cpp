#include <vksdl/device.hpp>
#include <vksdl/instance.hpp>
#include <vksdl/surface.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace vksdl {

Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
}

Device::Device(Device&& o) noexcept
    : device_(o.device_),
      physicalDevice_(o.physicalDevice_),
      graphicsQueue_(o.graphicsQueue_),
      presentQueue_(o.presentQueue_),
      transferQueue_(o.transferQueue_),
      families_(o.families_),
      minUboAlignment_(o.minUboAlignment_),
      maxMsaaSamples_(o.maxMsaaSamples_),
      timestampPeriod_(o.timestampPeriod_),
      gpuName_(std::move(o.gpuName_)),
      hasDeviceFault_(o.hasDeviceFault_),
      hasUnifiedLayouts_(o.hasUnifiedLayouts_),
      hasGPL_(o.hasGPL_),
      hasGplFastLinking_(o.hasGplFastLinking_),
      hasGplIndepInterp_(o.hasGplIndepInterp_),
      hasPCCC_(o.hasPCCC_),
      hasPushDescriptors_(o.hasPushDescriptors_),
      hasBindless_(o.hasBindless_),
      pfnTraceRays_(o.pfnTraceRays_),
      rtHandleSize_(o.rtHandleSize_),
      rtBaseAlignment_(o.rtBaseAlignment_),
      rtHandleAlignment_(o.rtHandleAlignment_),
      rtMaxRecursion_(o.rtMaxRecursion_),
      rtScratchAlignment_(o.rtScratchAlignment_) {
    o.device_         = VK_NULL_HANDLE;
    o.physicalDevice_ = VK_NULL_HANDLE;
    o.graphicsQueue_  = VK_NULL_HANDLE;
    o.presentQueue_   = VK_NULL_HANDLE;
    o.transferQueue_  = VK_NULL_HANDLE;
    o.families_       = {};
    o.pfnTraceRays_   = nullptr;
}

Device& Device::operator=(Device&& o) noexcept {
    if (this != &o) {
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
        }
        device_            = o.device_;
        physicalDevice_    = o.physicalDevice_;
        graphicsQueue_     = o.graphicsQueue_;
        presentQueue_      = o.presentQueue_;
        transferQueue_     = o.transferQueue_;
        families_          = o.families_;
        minUboAlignment_   = o.minUboAlignment_;
        maxMsaaSamples_    = o.maxMsaaSamples_;
        timestampPeriod_   = o.timestampPeriod_;
        gpuName_           = std::move(o.gpuName_);
        hasDeviceFault_    = o.hasDeviceFault_;
        hasUnifiedLayouts_ = o.hasUnifiedLayouts_;
        hasGPL_            = o.hasGPL_;
        hasGplFastLinking_ = o.hasGplFastLinking_;
        hasGplIndepInterp_ = o.hasGplIndepInterp_;
        hasPCCC_           = o.hasPCCC_;
        hasPushDescriptors_ = o.hasPushDescriptors_;
        hasBindless_        = o.hasBindless_;
        pfnTraceRays_      = o.pfnTraceRays_;
        rtHandleSize_      = o.rtHandleSize_;
        rtBaseAlignment_   = o.rtBaseAlignment_;
        rtHandleAlignment_ = o.rtHandleAlignment_;
        rtMaxRecursion_    = o.rtMaxRecursion_;
        rtScratchAlignment_ = o.rtScratchAlignment_;
        o.device_         = VK_NULL_HANDLE;
        o.physicalDevice_ = VK_NULL_HANDLE;
        o.graphicsQueue_  = VK_NULL_HANDLE;
        o.presentQueue_   = VK_NULL_HANDLE;
        o.transferQueue_  = VK_NULL_HANDLE;
        o.families_       = {};
        o.pfnTraceRays_   = nullptr;
    }
    return *this;
}

void Device::waitIdle() const {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

std::string Device::queryDeviceFault() const {
    if (!hasDeviceFault_ || device_ == VK_NULL_HANDLE) return {};

    auto pfn = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
        vkGetDeviceProcAddr(device_, "vkGetDeviceFaultInfoEXT"));
    if (!pfn) return {};

    VkDeviceFaultCountsEXT counts{};
    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    VkResult vr = pfn(device_, &counts, nullptr);
    if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) return {};

    std::vector<VkDeviceFaultAddressInfoEXT> addressInfos(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT>  vendorInfos(counts.vendorInfoCount);

    VkDeviceFaultInfoEXT info{};
    info.sType             = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    info.pAddressInfos     = addressInfos.empty() ? nullptr : addressInfos.data();
    info.pVendorInfos      = vendorInfos.empty() ? nullptr : vendorInfos.data();
    info.pVendorBinaryData = nullptr;

    vr = pfn(device_, &counts, &info);
    if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) return {};

    std::string result = "Device fault: ";
    result += info.description;

    for (std::uint32_t i = 0; i < counts.addressInfoCount; ++i) {
        result += "\n  address fault [" + std::to_string(i) + "]: type="
                + std::to_string(addressInfos[i].addressType)
                + ([&]() {
                    char hex[32];
                    std::snprintf(hex, sizeof(hex), " addr=0x%" PRIx64,
                                  static_cast<std::uint64_t>(addressInfos[i].reportedAddress));
                    return std::string(hex);
                })();
    }
    for (std::uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
        result += "\n  vendor fault [" + std::to_string(i) + "]: "
                + std::string(vendorInfos[i].description)
                + " code=" + std::to_string(vendorInfos[i].vendorFaultCode)
                + " data=" + std::to_string(vendorInfos[i].vendorFaultData);
    }

    return result;
}

DeviceBuilder::DeviceBuilder(const Instance& instance, const Surface& surface)
    : instance_(instance.vkInstance()), surface_(surface.vkSurface()) {}

DeviceBuilder& DeviceBuilder::needSwapchain() {
    return requireExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
}

DeviceBuilder& DeviceBuilder::needDynamicRendering() {
    return requireFeatures([](VkPhysicalDeviceVulkan13Features& f) {
        f.dynamicRendering = VK_TRUE;
    });
}

DeviceBuilder& DeviceBuilder::needSync2() {
    return requireFeatures([](VkPhysicalDeviceVulkan13Features& f) {
        f.synchronization2 = VK_TRUE;
    });
}

DeviceBuilder& DeviceBuilder::needGPL() {
    requireExtension(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
    requireExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    needGPL_ = true;
    return *this;
}

DeviceBuilder& DeviceBuilder::needRayTracingPipeline() {
    requireExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    requireExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    requireExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    requireFeatures12([](VkPhysicalDeviceVulkan12Features& f) {
        f.bufferDeviceAddress = VK_TRUE;
    });
    needRayTracingPipeline_    = true;
    needAccelerationStructure_ = true;
    return *this;
}

DeviceBuilder& DeviceBuilder::needRayQuery() {
    requireExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    requireExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    requireExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    requireFeatures12([](VkPhysicalDeviceVulkan12Features& f) {
        f.bufferDeviceAddress = VK_TRUE;
    });
    needRayQuery_              = true;
    needAccelerationStructure_ = true;
    return *this;
}

DeviceBuilder& DeviceBuilder::preferDiscreteGpu() {
    return preferGpu(GpuPrefer::Discrete);
}

DeviceBuilder& DeviceBuilder::preferIntegratedGpu() {
    return preferGpu(GpuPrefer::Integrated);
}

DeviceBuilder& DeviceBuilder::preferGpu(GpuPrefer pref) {
    gpuPref_ = pref;
    return *this;
}

DeviceBuilder& DeviceBuilder::requireExtension(const char* name) {
    extensions_.push_back(name);
    return *this;
}

DeviceBuilder& DeviceBuilder::requireFeatures(
    std::function<void(VkPhysicalDeviceVulkan13Features&)> configureFn) {
    featureRequests_.push_back({std::move(configureFn)});
    return *this;
}

DeviceBuilder& DeviceBuilder::requireFeatures12(
    std::function<void(VkPhysicalDeviceVulkan12Features&)> configureFn) {
    featureRequests12_.push_back({std::move(configureFn)});
    return *this;
}

DeviceBuilder& DeviceBuilder::requireCoreFeatures(
    std::function<void(VkPhysicalDeviceFeatures&)> configureFn) {
    coreFeatureRequests_.push_back({std::move(configureFn)});
    return *this;
}

DeviceBuilder& DeviceBuilder::chainFeatures(void* featureStruct) {
    chainedFeatures_.push_back(featureStruct);
    return *this;
}

QueueFamilies DeviceBuilder::findQueueFamilies(VkPhysicalDevice gpu) const {
    QueueFamilies result;

    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());

    for (std::uint32_t i = 0; i < count; ++i) {
        bool hasGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool hasTransfer = (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        bool hasCompute  = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)  != 0;

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface_, &presentSupport);

        if (hasGraphics && presentSupport) {
            result.graphics = i;
            result.present  = i;
        } else {
            if (hasGraphics && result.graphics == UINT32_MAX) {
                result.graphics = i;
            }
            if (presentSupport && result.present == UINT32_MAX) {
                result.present = i;
            }
        }

        // Prefer a transfer-only family (TRANSFER but not GRAPHICS or COMPUTE).
        // Fall back to transfer+compute (but not graphics) if no pure transfer.
        if (hasTransfer && !hasGraphics) {
            if (!hasCompute && result.transfer == UINT32_MAX) {
                result.transfer = i; // ideal: dedicated transfer
            } else if (hasCompute && result.transfer == UINT32_MAX) {
                result.transfer = i; // acceptable: async compute+transfer
            }
        }
    }

    return result;
}

bool DeviceBuilder::supportsExtensions(VkPhysicalDevice gpu) const {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, available.data());

    for (auto* required : extensions_) {
        bool found = false;
        for (auto& ext : available) {
            if (std::strcmp(ext.extensionName, required) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

int DeviceBuilder::scoreDevice(VkPhysicalDevice gpu) const {
    auto families = findQueueFamilies(gpu);
    if (!families.valid()) return -1;
    if (!supportsExtensions(gpu)) return -1;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);

    int score = 0;

    switch (gpuPref_) {
    case GpuPrefer::Discrete:
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += 100000;
        break;
    case GpuPrefer::Integrated:
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            score += 100000;
        break;
    case GpuPrefer::Any:
        break;
    }

    if (families.shared()) score += 100;

    // When RT was requested, prefer GPUs that support all required RT extensions.
    // supportsExtensions() already verified the base extensions are present (it
    // returned -1 above if not), so this bonus rewards GPUs that passed the check
    // over hypothetical future fallback paths.
    if (needAccelerationStructure_) {
        score += 5000;
    }

    // VRAM scoring: only count dedicated VRAM (DEVICE_LOCAL without HOST_VISIBLE).
    // Integrated GPUs report shared system RAM as DEVICE_LOCAL, but those heaps
    // have all memory types marked HOST_VISIBLE. Counting shared RAM inflates
    // integrated GPU scores above discrete GPUs with smaller dedicated VRAM.
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem);
    for (std::uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        if (!(mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
            continue;
        // A dedicated VRAM heap has at least one memory type that is NOT
        // host-visible. Shared system RAM heaps are fully host-visible.
        bool hasDedicatedType = false;
        for (std::uint32_t t = 0; t < mem.memoryTypeCount; ++t) {
            if (mem.memoryTypes[t].heapIndex == i &&
                !(mem.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                hasDedicatedType = true;
                break;
            }
        }
        if (hasDedicatedType) {
            score += static_cast<int>(mem.memoryHeaps[i].size / (1024 * 1024));
            break;
        }
    }

    return score;
}

Result<Device> DeviceBuilder::build() {
    std::uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance_, &gpuCount, nullptr);
    if (gpuCount == 0) {
        return Error{"select GPU", 0,
                     "No Vulkan-capable GPUs found.\n"
                     "Make sure you have a GPU with Vulkan driver support."};
    }

    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance_, &gpuCount, gpus.data());

    VkPhysicalDevice bestGpu = VK_NULL_HANDLE;
    int bestScore = -1;

    for (auto gpu : gpus) {
        int score = scoreDevice(gpu);
        if (score > bestScore) {
            bestScore = score;
            bestGpu   = gpu;
        }
    }

    if (bestGpu == VK_NULL_HANDLE) {
        std::string msg = "No suitable GPU found. Requirements:\n";
        for (auto* ext : extensions_) {
            msg += "  - extension: ";
            msg += ext;
            msg += "\n";
        }
        msg += "  - graphics + present queue support\n";
        msg += "Available GPUs:\n";
        for (auto gpu : gpus) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(gpu, &props);
            msg += "  - ";
            msg += props.deviceName;
            msg += " (score: ";
            msg += std::to_string(scoreDevice(gpu));
            msg += ")\n";
        }
        return Error{"select GPU", 0, msg};
    }

    auto families = findQueueFamilies(bestGpu);

    std::set<std::uint32_t> uniqueFamilies = {families.graphics, families.present};
    if (families.transfer != UINT32_MAX) {
        uniqueFamilies.insert(families.transfer);
    }
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    float priority = 1.0f;

    for (auto family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &priority;
        queueCIs.push_back(qci);
    }

    // pNext chain order: Features2 -> Vulkan12 -> Vulkan13 -> extension feature structs
    // -> user-chained structs. Core structs first, then extensions in request
    // order to ensure consistency for future extensions.

    // pipelineCreationCacheControl is always requested (Vulkan 1.3 core).
    // Even if the driver reports it as unsupported, Device tracks the result.
    bool needFeatures13 = true;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    // Always enable BDA. Every Vulkan 1.3 driver supports it, VMA assumes it
    // (VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT), and vertex/index buffer
    // convenience methods add SHADER_DEVICE_ADDRESS usage for RT readiness.
    features12.bufferDeviceAddress = VK_TRUE;
    // Timeline semaphores are core in 1.2 but must be explicitly enabled.
    // Required by TimelineSync and TransferQueue.
    features12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    for (auto& req : featureRequests12_) {
        req.configure(features12);
    }
    for (auto& req : featureRequests_) {
        req.configure(features13);
    }

    // Query physical device features for opportunistic feature detection.
    VkPhysicalDeviceVulkan12Features supported12{};
    supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features supported13{};
    supported13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    supported12.pNext = &supported13;
    VkPhysicalDeviceFeatures2 query{};
    query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    query.pNext = &supported12;
    vkGetPhysicalDeviceFeatures2(bestGpu, &query);

    bool havePCCC = (supported13.pipelineCreationCacheControl == VK_TRUE);
    if (havePCCC) {
        features13.pipelineCreationCacheControl = VK_TRUE;
    }

    // Bindless descriptor indexing (Vulkan 1.2 core features).
    // Enable opportunistically when the two essential features are supported.
    bool haveBindless =
        supported12.descriptorBindingPartiallyBound == VK_TRUE &&
        supported12.runtimeDescriptorArray          == VK_TRUE;
    if (haveBindless) {
        features12.descriptorBindingPartiallyBound = VK_TRUE;
        features12.runtimeDescriptorArray          = VK_TRUE;
        // Enable per-type updateAfterBind and nonUniformIndexing when supported.
        if (supported12.descriptorBindingSampledImageUpdateAfterBind)  {
            features12.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
            features12.shaderSampledImageArrayNonUniformIndexing     = VK_TRUE;
        }
        if (supported12.descriptorBindingStorageImageUpdateAfterBind) {
            features12.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
            features12.shaderStorageImageArrayNonUniformIndexing     = VK_TRUE;
        }
        if (supported12.descriptorBindingStorageBufferUpdateAfterBind) {
            features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
            features12.shaderStorageBufferArrayNonUniformIndexing    = VK_TRUE;
        }
        if (supported12.descriptorBindingUniformBufferUpdateAfterBind) {
            features12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
            features12.shaderUniformBufferArrayNonUniformIndexing    = VK_TRUE;
        }
    }

    // RT extension feature structs (only allocated when RT requested)
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{};
    rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqFeatures.rayQuery = VK_TRUE;

    // Enumerate available extensions for opportunistic feature detection.
    std::uint32_t availExtCount = 0;
    vkEnumerateDeviceExtensionProperties(bestGpu, nullptr, &availExtCount, nullptr);
    std::vector<VkExtensionProperties> availExts(availExtCount);
    vkEnumerateDeviceExtensionProperties(bestGpu, nullptr, &availExtCount, availExts.data());

    auto hasExtension = [&](const char* name) {
        for (auto& ext : availExts) {
            if (std::strcmp(ext.extensionName, name) == 0) return true;
        }
        return false;
    };

    // SER: enable opportunistically when the GPU supports it. Zero behavioral
    // impact -- the driver reorders shader invocations for better coherence.
    bool haveSer = needRayTracingPipeline_
                   && hasExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);

    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures{};
    serFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
    serFeatures.rayTracingInvocationReorder = VK_TRUE;

    // Device fault: enable opportunistically for better VK_ERROR_DEVICE_LOST
    // diagnostics. No behavioral impact when no fault occurs.
    bool haveDeviceFault = hasExtension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);

    // Unified image layouts: detect opportunistically. Extremely new extension.
    bool haveUnifiedLayouts = hasExtension("VK_KHR_unified_image_layouts");

    // Push descriptors: detect opportunistically.
    bool havePushDescriptors = hasExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    // Graphics pipeline library: detect opportunistically (or required via needGPL()).
    bool haveGPL = hasExtension(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);

    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gplFeatures{};
    gplFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
    gplFeatures.graphicsPipelineLibrary = VK_TRUE;

    VkPhysicalDeviceFaultFeaturesEXT faultFeatures{};
    faultFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
    faultFeatures.deviceFault = VK_TRUE;

    // Build the actual pNext chain: Features2 is the root.
    // We chain bottom-up by setting pNext pointers.
    // The tail of the chain gets linked first.
    void* pNextChain = nullptr;

    // User-chained structs go at the tail (last in chain = first linked)
    for (auto it = chainedFeatures_.rbegin(); it != chainedFeatures_.rend(); ++it) {
        auto* base = static_cast<VkBaseOutStructure*>(*it);
        base->pNext = static_cast<VkBaseOutStructure*>(pNextChain);
        pNextChain = *it;
    }

    if (haveGPL) {
        gplFeatures.pNext = pNextChain;
        pNextChain = &gplFeatures;
    }

    if (haveDeviceFault) {
        faultFeatures.pNext = pNextChain;
        pNextChain = &faultFeatures;
    }

    if (haveSer) {
        serFeatures.pNext = pNextChain;
        pNextChain = &serFeatures;
    }

    if (needRayQuery_) {
        rqFeatures.pNext = pNextChain;
        pNextChain = &rqFeatures;
    }
    if (needRayTracingPipeline_) {
        rtPipelineFeatures.pNext = pNextChain;
        pNextChain = &rtPipelineFeatures;
    }
    if (needAccelerationStructure_) {
        asFeatures.pNext = pNextChain;
        pNextChain = &asFeatures;
    }

    if (needFeatures13) {
        features13.pNext = pNextChain;
        pNextChain = &features13;
    }
    // features12 always chained (bufferDeviceAddress is always enabled).
    features12.pNext = pNextChain;
    pNextChain = &features12;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = pNextChain;

    for (auto& req : coreFeatureRequests_) {
        req.configure(features2.features);
    }

    // Add opportunistic extensions to the extension list
    std::vector<const char*> allExtensions = extensions_;
    if (haveSer) {
        allExtensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
    }
    if (haveDeviceFault) {
        allExtensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
    }
    if (havePushDescriptors) {
        allExtensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    }
    if (haveGPL) {
        // Avoid duplicate if user already called needGPL().
        bool alreadyInList = false;
        for (auto* ext : allExtensions) {
            if (std::strcmp(ext, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) == 0) {
                alreadyInList = true;
                break;
            }
        }
        if (!alreadyInList) {
            allExtensions.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
        }
        // VK_KHR_pipeline_library is a required dependency of VK_EXT_graphics_pipeline_library.
        bool havePipelineLib = false;
        for (auto* ext : allExtensions) {
            if (std::strcmp(ext, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) == 0) {
                havePipelineLib = true;
                break;
            }
        }
        if (!havePipelineLib) {
            allExtensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
        }
    }

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext                   = &features2;
    ci.queueCreateInfoCount    = static_cast<std::uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos       = queueCIs.data();
    ci.enabledExtensionCount   = static_cast<std::uint32_t>(allExtensions.size());
    ci.ppEnabledExtensionNames = allExtensions.data();

    Device dev;
    dev.physicalDevice_ = bestGpu;
    dev.families_       = families;

    VkResult vr = vkCreateDevice(bestGpu, &ci, nullptr, &dev.device_);
    if (vr != VK_SUCCESS) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(bestGpu, &props);
        std::string msg = "vkCreateDevice failed for '";
        msg += props.deviceName;
        msg += "'";
        if (vr == VK_ERROR_FEATURE_NOT_PRESENT) {
            msg += ".\nA requested Vulkan feature is not supported by this GPU.\n"
                   "Check that your GPU and driver support the features you requested.";
        }
        return Error{"create device", static_cast<std::int32_t>(vr), msg};
    }

    vkGetDeviceQueue(dev.device_, families.graphics, 0, &dev.graphicsQueue_);
    vkGetDeviceQueue(dev.device_, families.present,  0, &dev.presentQueue_);
    if (families.transfer != UINT32_MAX) {
        vkGetDeviceQueue(dev.device_, families.transfer, 0, &dev.transferQueue_);
    } else {
        dev.transferQueue_ = dev.graphicsQueue_;
    }

    dev.hasDeviceFault_      = haveDeviceFault;
    dev.hasUnifiedLayouts_   = haveUnifiedLayouts;
    dev.hasPCCC_             = havePCCC;
    dev.hasPushDescriptors_  = havePushDescriptors;
    dev.hasBindless_         = haveBindless;

    // Query GPL properties when the extension is enabled.
    if (haveGPL) {
        dev.hasGPL_ = true;

        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gplProps{};
        gplProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT;

        VkPhysicalDeviceProperties2 gplProps2{};
        gplProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        gplProps2.pNext = &gplProps;
        vkGetPhysicalDeviceProperties2(bestGpu, &gplProps2);

        dev.hasGplFastLinking_  = (gplProps.graphicsPipelineLibraryFastLinking == VK_TRUE);
        dev.hasGplIndepInterp_  = (gplProps.graphicsPipelineLibraryIndependentInterpolationDecoration == VK_TRUE);
    }

    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(bestGpu, &devProps);
    dev.minUboAlignment_  = devProps.limits.minUniformBufferOffsetAlignment;
    dev.timestampPeriod_  = devProps.limits.timestampPeriod;
    dev.gpuName_          = devProps.deviceName;

    if (needRayTracingPipeline_) {
        dev.pfnTraceRays_ = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(dev.device_, "vkCmdTraceRaysKHR"));
    }

    VkSampleCountFlags combined = devProps.limits.framebufferColorSampleCounts
                                & devProps.limits.framebufferDepthSampleCounts;
    for (VkSampleCountFlagBits bit : {
             VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT,
             VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_8_BIT,
             VK_SAMPLE_COUNT_4_BIT,  VK_SAMPLE_COUNT_2_BIT}) {
        if (combined & bit) {
            dev.maxMsaaSamples_ = bit;
            break;
        }
    }

    // Query RT properties when acceleration structures were requested
    if (needAccelerationStructure_) {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
        asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
        rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

        if (needRayTracingPipeline_) {
            rtProps.pNext = &asProps;
            props2.pNext  = &rtProps;
        } else {
            props2.pNext = &asProps;
        }

        vkGetPhysicalDeviceProperties2(bestGpu, &props2);

        dev.rtScratchAlignment_ = asProps.minAccelerationStructureScratchOffsetAlignment;

        if (needRayTracingPipeline_) {
            dev.rtHandleSize_      = rtProps.shaderGroupHandleSize;
            dev.rtBaseAlignment_   = rtProps.shaderGroupBaseAlignment;
            dev.rtHandleAlignment_ = rtProps.shaderGroupHandleAlignment;
            dev.rtMaxRecursion_    = rtProps.maxRayRecursionDepth;
        }
    }

    return dev;
}

} // namespace vksdl
