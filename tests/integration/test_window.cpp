#include <vksdl/app.hpp>
#include <vksdl/window.hpp>

#include <cassert>
#include <cstdio>

int main() {
    auto appResult = vksdl::App::create();
    assert(appResult.ok() && "App::create failed");
    auto app = std::move(appResult.value());

    auto result = app.createWindow("vksdl test", 640, 480);
    assert(result.ok() && "window creation failed");

    auto window = std::move(result.value());

    // SDL escape hatch should be valid
    assert(window.sdlWindow() != nullptr);

    // Pixel size should match requested size
    auto size = window.pixelSize();
    assert(size.width == 640);
    assert(size.height == 480);

    // No resize should have happened yet
    assert(!window.consumeResize());
    assert(!window.consumeResize());

    // Poll should return false (no events pending on a fresh window)
    vksdl::Event event{};
    while (window.pollEvent(event)) {
        // consume any startup events
    }

    std::printf("window test passed\n");
    return 0;
}
