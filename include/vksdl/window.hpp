#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

struct SDL_Window; // forward-declare -- no SDL.h in user code

namespace vksdl {

struct Size {
    std::uint32_t width  = 0;
    std::uint32_t height = 0;
};

enum class Key {
    Unknown,
    Escape,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    W,
    A,
    S,
    D,
    Space,
    LeftShift,
    Left,
    Right,
    Up,
    Down,
};

[[nodiscard]] Key keyFromScancode(int scancode);
[[nodiscard]] int scancodeFromKey(Key key);

enum class EventType {
    None,
    Quit,
    CloseRequested,
    Resized,
    KeyDown,
    KeyUp,
    MouseWheel,
    MouseButtonDown,
};

struct Event {
    EventType type = EventType::None;
    Size      size   = {};     // valid when type == Resized
    int       key    = 0;      // raw scancode (escape hatch)
    Key       keyCode = Key::Unknown; // typed key mapping
    float     scroll = 0.0f;   // valid when type == MouseWheel (positive = up)
    int       button = 0;      // valid when type == MouseButtonDown (1=L, 2=M, 3=R)
    int       clicks = 0;      // valid when type == MouseButtonDown (2 = double-click)
    float     mouseX = 0.0f;   // valid when type == MouseButtonDown (pixel coords)
    float     mouseY = 0.0f;   // valid when type == MouseButtonDown (pixel coords)
};

class App;
class WindowImpl;

class Window {
public:
    ~Window();
    Window(Window&&) noexcept;
    Window& operator=(Window&&) noexcept;
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Drain one event from this window's queue.
    // Returns true if an event was written, false when queue is empty.
    // Calls App::pumpEvents() internally on the first call per frame.
    bool pollEvent(Event& event);

    // Pixel size of the client area (accounts for DPI/HiDPI).
    [[nodiscard]] Size pixelSize() const;

    // Returns true once if the window was resized since the last call.
    [[nodiscard]] bool consumeResize();

    [[nodiscard]] Result<void> setTitle(std::string_view title);

    // Escape hatch -- typed pointer, forward-declared above.
    [[nodiscard]] SDL_Window* sdlWindow() const;

    // SDL window ID for event routing.
    [[nodiscard]] std::uint32_t windowId() const;

private:
    friend class App;
    explicit Window(std::unique_ptr<WindowImpl> impl, App* app);

    std::unique_ptr<WindowImpl> impl_;
    App* app_ = nullptr;
};

} // namespace vksdl
