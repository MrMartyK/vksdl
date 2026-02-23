#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

namespace vksdl {

class Window;
class AppImpl;

// SDL lifecycle owner. Create one App before any windows.
// Pumps the global SDL event queue and routes events to per-window queues.
class App {
public:
    [[nodiscard]] static Result<App> create();

    ~App();
    App(App&&) noexcept;
    App& operator=(App&&) noexcept;
    App(const App&) = delete;
    App& operator=(const App&) = delete;

    [[nodiscard]] Result<Window> createWindow(std::string_view title,
                                               std::uint32_t width,
                                               std::uint32_t height);

    void pumpEvents();

private:
    friend class Window;
    App() = default;
    void registerWindow(Window* w);
    void unregisterWindow(Window* w);
    void resetPump();

    std::unique_ptr<AppImpl> impl_;
};

} // namespace vksdl
