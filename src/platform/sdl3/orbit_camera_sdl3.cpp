#include <vksdl/orbit_camera.hpp>

#include <SDL3/SDL.h> // IWYU pragma: keep

#include <algorithm>
#include <cmath>
#include <numbers>

namespace vksdl {

static constexpr float kPi = std::numbers::pi_v<float>;
static constexpr float kPitchLimit = kPi * 0.49f;  // ~88.2 degrees

OrbitCamera::OrbitCamera(float targetX, float targetY, float targetZ,
                         float distance, float yaw, float pitch)
    : tgt_{targetX, targetY, targetZ},
      yaw_(yaw), pitch_(pitch), dist_(distance) {
    dist_ = std::max(dist_, minDist_);
    recomputeBasis();
}

void OrbitCamera::setTarget(float x, float y, float z) {
    tgt_[0] = x; tgt_[1] = y; tgt_[2] = z;
    recomputeBasis();
}

void OrbitCamera::setDistance(float d) {
    dist_ = std::clamp(d, minDist_, maxDist_);
    recomputeBasis();
}

void OrbitCamera::recomputeBasis() {
    float cy = std::cos(yaw_);
    float sy = std::sin(yaw_);
    float cp = std::cos(pitch_);
    float sp = std::sin(pitch_);

    // Forward: from eye toward target (same formula as FlyCamera).
    fwd_[0] = sy * cp;
    fwd_[1] = sp;
    fwd_[2] = cy * cp;

    // Right = normalize(cross(worldUp, fwd)).
    float rx = cy;
    float rz = -sy;
    float rLen = std::sqrt(rx * rx + rz * rz);
    if (rLen > 0.0001f) {
        right_[0] = rx / rLen;
        right_[1] = 0.0f;
        right_[2] = rz / rLen;
    }

    // Up = cross(fwd, right).
    up_[0] = fwd_[1] * right_[2] - fwd_[2] * right_[1];
    up_[1] = fwd_[2] * right_[0] - fwd_[0] * right_[2];
    up_[2] = fwd_[0] * right_[1] - fwd_[1] * right_[0];

    // Eye position: target - forward * distance.
    pos_[0] = tgt_[0] - fwd_[0] * dist_;
    pos_[1] = tgt_[1] - fwd_[1] * dist_;
    pos_[2] = tgt_[2] - fwd_[2] * dist_;
}

bool OrbitCamera::update(float dt) {
    quit_ = false;
    bool moved = false;
    (void)dt;  // orbit camera is purely input-driven, no velocity integration

    // Always drain mouse deltas to prevent accumulation spikes.
    float mx = 0.0f;
    float my = 0.0f;
    Uint32 buttons = SDL_GetRelativeMouseState(&mx, &my);

    // Consume scroll delta accumulated via feedScrollDelta().
    float scrollY = pendingScroll_;
    pendingScroll_ = 0.0f;

    // LMB: orbit (yaw/pitch).
    if ((buttons & SDL_BUTTON_LMASK) != 0) {
        if (mx != 0.0f || my != 0.0f) {
            yaw_   += mx * orbitSens_;
            pitch_ -= my * orbitSens_;
            pitch_ = std::clamp(pitch_, -kPitchLimit, kPitchLimit);
            moved = true;
        }
    }

    // MMB or RMB: pan (translate target in camera right/up plane).
    if ((buttons & (SDL_BUTTON_MMASK | SDL_BUTTON_RMASK)) != 0) {
        if (mx != 0.0f || my != 0.0f) {
            // World units per pixel at target depth.
            float spanY = 2.0f * dist_ * std::tan(fovY_ * 0.5f);
            float wuPerPx = spanY / static_cast<float>(vpH_);
            float panScale = wuPerPx * panSens_;

            tgt_[0] += (-right_[0] * mx + up_[0] * my) * panScale;
            tgt_[1] += (-right_[1] * mx + up_[1] * my) * panScale;
            tgt_[2] += (-right_[2] * mx + up_[2] * my) * panScale;
            moved = true;
        }
    }

    // Scroll: zoom (exponential for scale-invariant feel).
    if (scrollY != 0.0f) {
        dist_ *= std::exp(-scrollY * zoomSens_);
        dist_ = std::clamp(dist_, minDist_, maxDist_);
        moved = true;
    }

    // Keyboard.
    const bool* keys = SDL_GetKeyboardState(nullptr);
    if (keys[SDL_SCANCODE_ESCAPE]) {
        quit_ = true;
    }

    recomputeBasis();
    return moved;
}

} // namespace vksdl
