#pragma once

// Internal header -- shared between app_sdl3.cpp and window_sdl3.cpp.
// Not part of the public API.

#include <vksdl/window.hpp>

#include <SDL3/SDL.h>

#include <queue>

namespace vksdl {

class WindowImpl {
public:
    SDL_Window*        sdlWindow = nullptr;
    SDL_WindowID       windowId  = 0;
    bool               resized   = false;
    std::queue<Event>  events;
};

} // namespace vksdl
