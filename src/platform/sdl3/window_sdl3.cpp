#include "window_impl.hpp"
#include <vksdl/app.hpp>

namespace vksdl {

Window::Window(std::unique_ptr<WindowImpl> impl, App* app)
    : impl_(std::move(impl)), app_(app) {
    if (app_) app_->registerWindow(this);
}

Window::~Window() {
    if (app_) app_->unregisterWindow(this);
    if (impl_ && impl_->sdlWindow) {
        SDL_DestroyWindow(impl_->sdlWindow);
    }
}

Window::Window(Window&& o) noexcept
    : impl_(std::move(o.impl_)), app_(o.app_) {
    o.app_ = nullptr;
    if (app_) {
        app_->unregisterWindow(&o);
        app_->registerWindow(this);
    }
}

Window& Window::operator=(Window&& o) noexcept {
    if (this != &o) {
        if (app_) app_->unregisterWindow(this);
        if (impl_ && impl_->sdlWindow) {
            SDL_DestroyWindow(impl_->sdlWindow);
        }

        impl_ = std::move(o.impl_);
        app_  = o.app_;
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
        if (app_) app_->resetPump();
        return false;
    }

    event = impl_->events.front();
    impl_->events.pop();
    return true;
}

Size Window::pixelSize() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(impl_->sdlWindow, &w, &h);
    return Size{
        static_cast<std::uint32_t>(w),
        static_cast<std::uint32_t>(h)
    };
}

bool Window::consumeResize() {
    bool r = impl_->resized;
    impl_->resized = false;
    return r;
}

SDL_Window* Window::sdlWindow() const {
    return impl_->sdlWindow;
}

std::uint32_t Window::windowId() const {
    return impl_->windowId;
}

} // namespace vksdl
