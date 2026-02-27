#pragma once

#include <cstdint>

namespace vksdl {

// First-person fly camera with mouse-look and WASD movement.
// Reads SDL mouse/keyboard state internally -- public header has no SDL dependency.
// Usage:
//   FlyCamera camera(13, 2, 3, 0.2f, -0.15f);
//   bool moved = camera.update(dt);  // call once per frame
//   camera.position();               // float[3]
//   camera.forward();                // float[3], etc.
//
// Thread safety: thread-confined (main/UI thread).
class FlyCamera {
  public:
    FlyCamera(float x, float y, float z, float yaw = 0.0f, float pitch = 0.0f);

    // Process mouse + keyboard input. Returns true if the camera moved.
    // Right mouse button must be held for mouse-look.
    // WASD = move, Space = up, LShift = down, Escape = sets shouldQuit().
    [[nodiscard]] bool update(float dt);

    // True if Escape was pressed during the last update().
    [[nodiscard]] bool shouldQuit() const {
        return quit_;
    }

    void setSpeed(float moveSpeed) {
        moveSpeed_ = moveSpeed;
    }
    void setLookSensitivity(float sens) {
        lookSens_ = sens;
    }

    [[nodiscard]] const float* position() const {
        return pos_;
    }
    [[nodiscard]] const float* forward() const {
        return fwd_;
    }
    [[nodiscard]] const float* right() const {
        return right_;
    }
    [[nodiscard]] const float* up() const {
        return up_;
    }

    [[nodiscard]] float yaw() const {
        return yaw_;
    }
    [[nodiscard]] float pitch() const {
        return pitch_;
    }

  private:
    void recomputeBasis();

    float pos_[3];
    float fwd_[3] = {0, 0, 1};
    float right_[3] = {1, 0, 0};
    float up_[3] = {0, 1, 0};
    float yaw_;
    float pitch_;
    float moveSpeed_ = 5.0f;
    float lookSens_ = 0.003f;
    bool quit_ = false;
};

} // namespace vksdl
