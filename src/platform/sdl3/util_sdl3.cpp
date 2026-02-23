#include <vksdl/util.hpp>

#include <SDL3/SDL.h>

#include <filesystem>

namespace vksdl {

std::filesystem::path exeDir() {
    return std::filesystem::path(SDL_GetBasePath());
}

std::filesystem::path exeRelativePath(const std::filesystem::path& relative) {
    return exeDir() / relative;
}

} // namespace vksdl
