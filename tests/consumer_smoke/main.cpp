#include <vksdl/vksdl.hpp>

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <string>

int main() {
    vksdl::Error err{"consumer_smoke", 0, "ok"};
    const std::string formatted = err.format();
    if (formatted.empty()) {
        return 1;
    }

    const int sdl_ver = SDL_GetVersion();
    const int sdl_major = SDL_VERSIONNUM_MAJOR(sdl_ver);
    const int sdl_minor = SDL_VERSIONNUM_MINOR(sdl_ver);
    const int sdl_micro = SDL_VERSIONNUM_MICRO(sdl_ver);

    std::uint32_t api_version = 0;
    const auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerate_instance_version != nullptr) {
        (void)enumerate_instance_version(&api_version);
    }

    std::printf(
        "vksdl consumer smoke ok | SDL %d.%d.%d | Vulkan 0x%08x\n",
        sdl_major, sdl_minor, sdl_micro, api_version);
    return 0;
}
