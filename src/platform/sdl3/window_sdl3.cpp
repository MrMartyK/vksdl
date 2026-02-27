#include "window_impl.hpp"
#include <vksdl/app.hpp>

#include <string>

namespace vksdl {

Key keyFromScancode(int scancode) {
    switch (scancode) {
    case SDL_SCANCODE_ESCAPE:
        return Key::Escape;
    case SDL_SCANCODE_1:
        return Key::Digit1;
    case SDL_SCANCODE_2:
        return Key::Digit2;
    case SDL_SCANCODE_3:
        return Key::Digit3;
    case SDL_SCANCODE_4:
        return Key::Digit4;
    case SDL_SCANCODE_5:
        return Key::Digit5;
    case SDL_SCANCODE_6:
        return Key::Digit6;
    case SDL_SCANCODE_7:
        return Key::Digit7;
    case SDL_SCANCODE_8:
        return Key::Digit8;
    case SDL_SCANCODE_9:
        return Key::Digit9;
    case SDL_SCANCODE_W:
        return Key::W;
    case SDL_SCANCODE_A:
        return Key::A;
    case SDL_SCANCODE_S:
        return Key::S;
    case SDL_SCANCODE_D:
        return Key::D;
    case SDL_SCANCODE_SPACE:
        return Key::Space;
    case SDL_SCANCODE_LSHIFT:
        return Key::LeftShift;
    case SDL_SCANCODE_LEFT:
        return Key::Left;
    case SDL_SCANCODE_RIGHT:
        return Key::Right;
    case SDL_SCANCODE_UP:
        return Key::Up;
    case SDL_SCANCODE_DOWN:
        return Key::Down;
    default:
        return Key::Unknown;
    }
}

int scancodeFromKey(Key key) {
    switch (key) {
    case Key::Escape:
        return SDL_SCANCODE_ESCAPE;
    case Key::Digit1:
        return SDL_SCANCODE_1;
    case Key::Digit2:
        return SDL_SCANCODE_2;
    case Key::Digit3:
        return SDL_SCANCODE_3;
    case Key::Digit4:
        return SDL_SCANCODE_4;
    case Key::Digit5:
        return SDL_SCANCODE_5;
    case Key::Digit6:
        return SDL_SCANCODE_6;
    case Key::Digit7:
        return SDL_SCANCODE_7;
    case Key::Digit8:
        return SDL_SCANCODE_8;
    case Key::Digit9:
        return SDL_SCANCODE_9;
    case Key::W:
        return SDL_SCANCODE_W;
    case Key::A:
        return SDL_SCANCODE_A;
    case Key::S:
        return SDL_SCANCODE_S;
    case Key::D:
        return SDL_SCANCODE_D;
    case Key::Space:
        return SDL_SCANCODE_SPACE;
    case Key::LeftShift:
        return SDL_SCANCODE_LSHIFT;
    case Key::Left:
        return SDL_SCANCODE_LEFT;
    case Key::Right:
        return SDL_SCANCODE_RIGHT;
    case Key::Up:
        return SDL_SCANCODE_UP;
    case Key::Down:
        return SDL_SCANCODE_DOWN;
    case Key::Unknown:
        return 0;
    }
    return 0;
}

Window::Window(std::unique_ptr<WindowImpl> impl, App* app) : impl_(std::move(impl)), app_(app) {
    if (app_)
        app_->registerWindow(this);
}

Window::~Window() {
    if (app_)
        app_->unregisterWindow(this);
    if (impl_ && impl_->sdlWindow) {
        SDL_DestroyWindow(impl_->sdlWindow);
    }
}

Window::Window(Window&& o) noexcept : impl_(std::move(o.impl_)), app_(o.app_) {
    o.app_ = nullptr;
    if (app_) {
        app_->unregisterWindow(&o);
        app_->registerWindow(this);
    }
}

Window& Window::operator=(Window&& o) noexcept {
    if (this != &o) {
        if (app_)
            app_->unregisterWindow(this);
        if (impl_ && impl_->sdlWindow) {
            SDL_DestroyWindow(impl_->sdlWindow);
        }

        impl_ = std::move(o.impl_);
        app_ = o.app_;
        o.app_ = nullptr;

        if (app_) {
            app_->unregisterWindow(&o);
            app_->registerWindow(this);
        }
    }
    return *this;
}

bool Window::pollEvent(Event& event) {
    if (!impl_) {
        event.type = EventType::None;
        return false;
    }

    // Pump once when the queue is empty (start of a new frame).
    // While draining queued events, skip re-pumping.
    if (impl_->events.empty() && app_) {
        app_->pumpEvents();
    }

    if (impl_->events.empty()) {
        event.type = EventType::None;
        // Reset pump flag so the next pollEvent call (next frame) will pump again.
        if (app_)
            app_->resetPump();
        return false;
    }

    event = impl_->events.front();
    impl_->events.pop();
    return true;
}

Size Window::pixelSize() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(impl_->sdlWindow, &w, &h);
    return Size{static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
}

bool Window::consumeResize() {
    bool r = impl_->resized;
    impl_->resized = false;
    return r;
}

Result<void> Window::setTitle(std::string_view title) {
    if (!impl_ || !impl_->sdlWindow) {
        return Error{"set window title", 0, "window is not initialized"};
    }

    std::string titleStr(title);
    if (!SDL_SetWindowTitle(impl_->sdlWindow, titleStr.c_str())) {
        return Error{"set window title", 0, SDL_GetError()};
    }
    return {};
}

SDL_Window* Window::sdlWindow() const {
    return impl_->sdlWindow;
}

std::uint32_t Window::windowId() const {
    return impl_->windowId;
}

} // namespace vksdl
