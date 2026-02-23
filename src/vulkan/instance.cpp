#include <vksdl/instance.hpp>
#include <vksdl/vulkan_wsi.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vksdl {

static const char* VkResultToString(VkResult r) {
    switch (r) {
    case VK_SUCCESS:                        return "success";
    case VK_NOT_READY:                      return "not ready";
    case VK_TIMEOUT:                        return "timeout";
    case VK_INCOMPLETE:                     return "incomplete";
    case VK_ERROR_OUT_OF_HOST_MEMORY:       return "out of host memory";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "out of GPU memory";
    case VK_ERROR_INITIALIZATION_FAILED:    return "initialization failed";
    case VK_ERROR_DEVICE_LOST:              return "device lost (GPU crashed or was removed)";
    case VK_ERROR_MEMORY_MAP_FAILED:        return "memory map failed";
    case VK_ERROR_LAYER_NOT_PRESENT:        return "requested layer not present";
    case VK_ERROR_EXTENSION_NOT_PRESENT:    return "requested extension not present";
    case VK_ERROR_FEATURE_NOT_PRESENT:      return "requested feature not present";
    case VK_ERROR_INCOMPATIBLE_DRIVER:      return "incompatible Vulkan driver";
    case VK_ERROR_TOO_MANY_OBJECTS:         return "too many objects";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "format not supported";
    case VK_ERROR_SURFACE_LOST_KHR:         return "surface lost";
    case VK_ERROR_OUT_OF_DATE_KHR:          return "swapchain out of date";
    default:
        return "unknown error";
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*userData*/)
{
    const char* level = "INFO";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        level = "ERROR";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        level = "WARN";

    std::fprintf(stderr, "vksdl [%s]: %s\n", level, data->pMessage);
    return VK_FALSE;
}

Instance::~Instance() {
    if (messenger_ != VK_NULL_HANDLE) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (func) func(instance_, messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

Instance::Instance(Instance&& o) noexcept
    : instance_(o.instance_), messenger_(o.messenger_) {
    o.instance_  = VK_NULL_HANDLE;
    o.messenger_ = VK_NULL_HANDLE;
}

Instance& Instance::operator=(Instance&& o) noexcept {
    if (this != &o) {
        if (messenger_ != VK_NULL_HANDLE) {
            auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (func) func(instance_, messenger_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
        instance_    = o.instance_;
        messenger_   = o.messenger_;
        o.instance_  = VK_NULL_HANDLE;
        o.messenger_ = VK_NULL_HANDLE;
    }
    return *this;
}

static bool LayerAvailable(const char* name,
                           const std::vector<VkLayerProperties>& available) {
    for (auto& layer : available) {
        if (std::strcmp(layer.layerName, name) == 0) return true;
    }
    return false;
}

static bool ExtensionAvailable(const char* name,
                               const std::vector<VkExtensionProperties>& available) {
    for (auto& ext : available) {
        if (std::strcmp(ext.extensionName, name) == 0) return true;
    }
    return false;
}

static void Deduplicate(std::vector<const char*>& list) {
    std::vector<const char*> unique;
    unique.reserve(list.size());
    for (auto* name : list) {
        bool found = false;
        for (auto* existing : unique) {
            if (std::strcmp(existing, name) == 0) { found = true; break; }
        }
        if (!found) unique.push_back(name);
    }
    list = std::move(unique);
}

InstanceBuilder& InstanceBuilder::appName(std::string_view name) {
    appName_ = name;
    return *this;
}

InstanceBuilder& InstanceBuilder::requireVulkan(std::uint32_t major,
                                                std::uint32_t minor,
                                                std::uint32_t patch) {
    apiVersion_ = VK_MAKE_API_VERSION(0, major, minor, patch);
    return *this;
}

InstanceBuilder& InstanceBuilder::validation(Validation v) {
    validation_ = v;
    return *this;
}

InstanceBuilder& InstanceBuilder::addExtension(const char* name) {
    extensions_.push_back(name);
    return *this;
}

InstanceBuilder& InstanceBuilder::addLayer(const char* name) {
    layers_.push_back(name);
    return *this;
}

InstanceBuilder& InstanceBuilder::enableWindowSupport() {
    windowSupport_ = true;
    return *this;
}

Result<Instance> InstanceBuilder::build() {
    std::vector<const char*> extensions = extensions_;

    if (windowSupport_) {
        auto wsi = wsi::requiredInstanceExtensions();
        extensions.insert(extensions.end(), wsi.begin(), wsi.end());
    }

    bool wantValidation = (validation_ == Validation::On);
    if (wantValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    Deduplicate(extensions);

    std::uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availableExts.data());

    for (auto* requested : extensions) {
        if (!ExtensionAvailable(requested, availableExts)) {
            return Error{"create instance", 0,
                         "Extension '" + std::string(requested) + "' is not available.\n"
                         "This usually means your GPU driver doesn't support it.\n"
                         "Try updating your GPU drivers, or check https://vulkan.gpuinfo.org"};
        }
    }

    std::vector<const char*> layers = layers_;
    if (wantValidation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    Deduplicate(layers);

    std::uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    bool hasValidationLayer = true;
    for (auto* requested : layers) {
        if (!LayerAvailable(requested, availableLayers)) {
            if (std::strcmp(requested, "VK_LAYER_KHRONOS_validation") == 0) {
                hasValidationLayer = false;
            } else {
                return Error{"create instance", 0,
                             "Layer '" + std::string(requested) + "' is not available.\n"
                             "Make sure the Vulkan SDK is installed."};
            }
        }
    }

    // Degrade gracefully when the validation layer is absent (release driver, CI).
    if (wantValidation && !hasValidationLayer) {
        wantValidation = false;
        std::erase_if(layers, [](const char* n) {
            return std::strcmp(n, "VK_LAYER_KHRONOS_validation") == 0;
        });
        std::erase_if(extensions, [](const char* n) {
            return std::strcmp(n, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        });
    }

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = appName_.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion         = apiVersion_;

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = static_cast<std::uint32_t>(layers.size());
    ci.ppEnabledLayerNames     = layers.data();

    // Chain the debug messenger into instance create info so validation fires
    // during vkCreateInstance/vkDestroyInstance (before the persistent messenger exists).
    VkDebugUtilsMessengerCreateInfoEXT debugCI{};
    if (wantValidation) {
        debugCI.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCI.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCI.pfnUserCallback = DebugCallback;
        ci.pNext = &debugCI;
    }

    Instance inst;
    VkResult vr = vkCreateInstance(&ci, nullptr, &inst.instance_);
    if (vr != VK_SUCCESS) {
        std::string msg = std::string(VkResultToString(vr));
        if (vr == VK_ERROR_INCOMPATIBLE_DRIVER) {
            std::uint32_t major = VK_API_VERSION_MAJOR(apiVersion_);
            std::uint32_t minor = VK_API_VERSION_MINOR(apiVersion_);
            msg += ".\nYou requested Vulkan " + std::to_string(major) + "." +
                   std::to_string(minor) + " but your driver doesn't support it.\n"
                   "Try updating your GPU drivers, or check https://vulkan.gpuinfo.org";
        }
        return Error{"create instance", static_cast<std::int32_t>(vr), msg};
    }

    if (wantValidation) {
        auto createFunc = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(inst.instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (createFunc) {
            VkDebugUtilsMessengerCreateInfoEXT messengerCI{};
            messengerCI.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            messengerCI.messageSeverity = debugCI.messageSeverity;
            messengerCI.messageType     = debugCI.messageType;
            messengerCI.pfnUserCallback = DebugCallback;
            createFunc(inst.instance_, &messengerCI, nullptr, &inst.messenger_);
        }
    }

    return inst;
}

} // namespace vksdl
