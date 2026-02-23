#include <vksdl/fly_camera.hpp>

#include <SDL3/SDL.h>

#include <cmath>

namespace vksdl {

static constexpr float kPi = 3.14159265358979323846f;

FlyCamera::FlyCamera(float x, float y, float z, float yaw, float pitch)
    : pos_{x, y, z}, yaw_(yaw), pitch_(pitch) {
    recomputeBasis();
}

void FlyCamera::recomputeBasis() {
    float cy = std::cos(yaw_),   sy = std::sin(yaw_);
    float cp = std::cos(pitch_), sp = std::sin(pitch_);

    fwd_[0]   = sy * cp;
    fwd_[1]   = sp;
    fwd_[2]   = cy * cp;

    // world up = {0, 1, 0}
    // right = normalize(cross(worldUp, fwd))
    float rx = cy;   // cross(worldUp, fwd) = (cy*cp, 0, -sy*cp) but we only need xz direction
    float rz = -sy;
    float rLen = std::sqrt(rx * rx + rz * rz);
    if (rLen > 0.0001f) {
        right_[0] = rx / rLen;
        right_[1] = 0.0f;
        right_[2] = rz / rLen;
    }

    // up = cross(fwd, right)
    up_[0] = fwd_[1] * right_[2] - fwd_[2] * right_[1];
    up_[1] = fwd_[2] * right_[0] - fwd_[0] * right_[2];
    up_[2] = fwd_[0] * right_[1] - fwd_[1] * right_[0];
}

bool FlyCamera::update(float dt) {
    quit_ = false;
    bool moved = false;

    // Mouse look (right mouse button held).
    float mx = 0.0f, my = 0.0f;
    Uint32 buttons = SDL_GetRelativeMouseState(&mx, &my);
    if (buttons & SDL_BUTTON_RMASK) {
        if (mx != 0.0f || my != 0.0f) {
            yaw_   += mx * lookSens_;
            pitch_ -= my * lookSens_;

            if (pitch_ >  kPi * 0.49f) pitch_ =  kPi * 0.49f;
            if (pitch_ < -kPi * 0.49f) pitch_ = -kPi * 0.49f;

            moved = true;
        }
    }

    recomputeBasis();

    // Keyboard movement.
    const bool* keys = SDL_GetKeyboardState(nullptr);
    if (keys[SDL_SCANCODE_ESCAPE]) {
        quit_ = true;
    }

    // Flat forward/right (XZ plane, no pitch component).
    float sy = std::sin(yaw_), cy = std::cos(yaw_);
    float flatFwdLen = std::sqrt(sy * sy + cy * cy);
    float ffx = sy / flatFwdLen, ffz = cy / flatFwdLen;
    float frx = cy / flatFwdLen, frz = -sy / flatFwdLen;

    float moveX = 0.0f, moveY = 0.0f, moveZ = 0.0f;
    if (keys[SDL_SCANCODE_W])      { moveX += ffx; moveZ += ffz; }
    if (keys[SDL_SCANCODE_S])      { moveX -= ffx; moveZ -= ffz; }
    if (keys[SDL_SCANCODE_D])      { moveX += frx; moveZ += frz; }
    if (keys[SDL_SCANCODE_A])      { moveX -= frx; moveZ -= frz; }
    if (keys[SDL_SCANCODE_SPACE])  { moveY += 1.0f; }
    if (keys[SDL_SCANCODE_LSHIFT]) { moveY -= 1.0f; }

    float moveLen = std::sqrt(moveX * moveX + moveY * moveY + moveZ * moveZ);
    if (moveLen > 0.001f) {
        float s = moveSpeed_ * dt / moveLen;
        pos_[0] += moveX * s;
        pos_[1] += moveY * s;
        pos_[2] += moveZ * s;
        moved = true;
    }

    return moved;
}

} // namespace vksdl
