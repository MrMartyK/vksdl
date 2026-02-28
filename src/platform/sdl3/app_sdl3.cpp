#include "window_impl.hpp"
#include <vksdl/app.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace vksdl {

class AppImpl {
  public:
    std::vector<Window*> windows; // non-owning, for event routing
    bool pumped = false;          // true after pumpEvents(), reset by resetPump()
};

Result<App> App::create() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        return Error{"initialize SDL", 0, std::string("SDL_Init failed: ") + SDL_GetError()};
    }
    App app;
    app.impl_ = std::make_unique<AppImpl>();
    return app;
}

App::~App() {
    if (impl_) {
        impl_.reset();
        SDL_Quit();
    }
}

App::App(App&&) noexcept = default;
App& App::operator=(App&&) noexcept = default;

Result<Window> App::createWindow(std::string_view title, std::uint32_t width,
                                 std::uint32_t height) {
    std::string titleStr(title);

    SDL_Window* sdlWin =
        SDL_CreateWindow(titleStr.c_str(), static_cast<int>(width), static_cast<int>(height),
                         SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    if (!sdlWin) {
        return Error{"create window", 0, std::string("SDL_CreateWindow failed: ") + SDL_GetError()};
    }

    auto impl = std::make_unique<WindowImpl>();
    impl->sdlWindow = sdlWin;
    impl->windowId = SDL_GetWindowID(sdlWin);

    return Window(std::move(impl), this);
}

void App::pumpEvents() {
    if (impl_->pumped)
        return;
    impl_->pumped = true;

    SDL_Event sdlEvent;
    while (SDL_PollEvent(&sdlEvent)) {
        switch (sdlEvent.type) {
        case SDL_EVENT_QUIT:
            for (auto* w : impl_->windows) {
                Event e{};
                e.type = EventType::Quit;
                w->impl_->events.push(e);
            }
            break;

        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            for (auto* w : impl_->windows) {
                if (w->windowId() == sdlEvent.window.windowID) {
                    Event e{};
                    e.type = EventType::CloseRequested;
                    w->impl_->events.push(e);
                }
            }
            break;

        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            for (auto* w : impl_->windows) {
                if (w->windowId() == sdlEvent.window.windowID && sdlEvent.window.data1 > 0 &&
                    sdlEvent.window.data2 > 0) {
                    Event e{};
                    e.type = EventType::Resized;
                    e.size.width = static_cast<std::uint32_t>(sdlEvent.window.data1);
                    e.size.height = static_cast<std::uint32_t>(sdlEvent.window.data2);
                    w->impl_->events.push(e);
                    w->impl_->resized = true;
                }
            }
            break;

        case SDL_EVENT_KEY_DOWN:
            for (auto* w : impl_->windows) {
                if (w->windowId() == sdlEvent.key.windowID) {
                    Event e{};
                    e.type = EventType::KeyDown;
                    e.key = static_cast<int>(sdlEvent.key.scancode);
                    e.keyCode = keyFromScancode(e.key);
                    w->impl_->events.push(e);
                }
            }
            break;

        case SDL_EVENT_KEY_UP:
            for (auto* w : impl_->windows) {
                if (w->windowId() == sdlEvent.key.windowID) {
                    Event e{};
                    e.type = EventType::KeyUp;
                    e.key = static_cast<int>(sdlEvent.key.scancode);
                    e.keyCode = keyFromScancode(e.key);
                    w->impl_->events.push(e);
                }
            }
            break;

        case SDL_EVENT_MOUSE_WHEEL:
            for (auto* w : impl_->windows) {
                if (w->windowId() == sdlEvent.wheel.windowID) {
                    Event e{};
                    e.type = EventType::MouseWheel;
                    e.scroll = sdlEvent.wheel.y;
                    w->impl_->events.push(e);
                }
            }
            break;

        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            for (auto* w : impl_->windows) {
                if (w->windowId() == sdlEvent.button.windowID) {
                    Event e{};
                    e.type = EventType::MouseButtonDown;
                    e.button = static_cast<int>(sdlEvent.button.button);
                    e.clicks = static_cast<int>(sdlEvent.button.clicks);
                    e.mouseX = sdlEvent.button.x;
                    e.mouseY = sdlEvent.button.y;
                    w->impl_->events.push(e);
                }
            }
            break;

        default:
            break;
        }
    }
}

void App::registerWindow(Window* w) {
    impl_->windows.push_back(w);
}

void App::unregisterWindow(Window* w) {
    auto& v = impl_->windows;
    v.erase(std::remove(v.begin(), v.end(), w), v.end());
}

void App::resetPump() {
    impl_->pumped = false;
}

} // namespace vksdl
