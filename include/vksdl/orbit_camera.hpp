#pragma once

#include <cstdint>

namespace vksdl {

// Orbit/turntable camera for scene viewers and editors.
// Reads SDL mouse/keyboard state internally -- public header has no SDL dependency.
// Usage:
//   OrbitCamera camera(0, 0.5f, 0,  5.0f);  // target, distance
//   camera.setViewport(1280, 720);
//   bool moved = camera.update(dt);
//   camera.position();  // float[3] -- derived eye position
//
// Input: LMB drag = orbit, MMB/RMB drag = pan, scroll = zoom, Escape = quit.
// Scroll input must be fed from the app's event loop via feedScrollDelta().
//
// Thread safety: thread-confined (main/UI thread).
class OrbitCamera {
public:
    OrbitCamera(float targetX, float targetY, float targetZ,
                float distance,
                float yaw = 0.0f, float pitch = 0.0f);

    // Process mouse + keyboard input. Returns true if the camera moved.
    [[nodiscard]] bool update(float dt);

    [[nodiscard]] bool shouldQuit() const { return quit_; }

    void setViewport(std::uint32_t w, std::uint32_t h) { vpW_ = w; vpH_ = h; }
    void setFovY(float fovYRadians) { fovY_ = fovYRadians; }

    void setDistanceLimits(float minD, float maxD) { minDist_ = minD; maxDist_ = maxD; }
    void setOrbitSensitivity(float s) { orbitSens_ = s; }
    void setPanSensitivity(float s)   { panSens_ = s; }
    void setZoomSensitivity(float s)  { zoomSens_ = s; }

    void setTarget(float x, float y, float z);
    void setDistance(float d);

    // Feed scroll wheel delta from the application's event loop.
    // Positive = zoom in, negative = zoom out (SDL convention).
    // Accumulated internally and consumed on the next update().
    void feedScrollDelta(float dy) { pendingScroll_ += dy; }

    [[nodiscard]] const float* position() const { return pos_; }
    [[nodiscard]] const float* target()   const { return tgt_; }
    [[nodiscard]] const float* forward()  const { return fwd_; }
    [[nodiscard]] const float* right()    const { return right_; }
    [[nodiscard]] const float* up()       const { return up_; }

    [[nodiscard]] float yaw()      const { return yaw_; }
    [[nodiscard]] float pitch()    const { return pitch_; }
    [[nodiscard]] float distance() const { return dist_; }

private:
    void recomputeBasis();

    float tgt_[3];
    float pos_[3]   = {0, 0, 0};
    float fwd_[3]   = {0, 0, 1};
    float right_[3] = {1, 0, 0};
    float up_[3]    = {0, 1, 0};

    float yaw_;
    float pitch_;
    float dist_;

    float orbitSens_ = 0.003f;
    float panSens_   = 1.0f;
    float zoomSens_  = 0.15f;

    float minDist_ = 0.05f;
    float maxDist_ = 1e6f;

    float fovY_ = 1.0471976f;  // 60 deg
    std::uint32_t vpW_ = 1;
    std::uint32_t vpH_ = 1;

    float pendingScroll_ = 0.0f;
    bool quit_ = false;
};

} // namespace vksdl
