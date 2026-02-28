#pragma once

#include <array>
#include <cmath>
#include <limits>

namespace vksdl {

// Column-major 4x4 matrix. Index with m[col*4 + row].
// Minimal -- just enough for projection/view helpers. Not a general math library.
struct Mat4 {
    std::array<float, 16> m{};

    float operator[](int i) const {
        return m[i];
    }
    float& operator[](int i) {
        return m[i];
    }

    // Column-major accessors: element at (row, col).
    float at(int row, int col) const {
        return m[col * 4 + row];
    }
    float& at(int row, int col) {
        return m[col * 4 + row];
    }

    // Raw pointer for glUniformMatrix4fv / vkCmdPushConstants.
    [[nodiscard]] const float* data() const {
        return m.data();
    }
};

// Identity matrix.
[[nodiscard]] inline Mat4 mat4Identity() {
    Mat4 out{};
    out.m[0] = 1.0f;
    out.m[5] = 1.0f;
    out.m[10] = 1.0f;
    out.m[15] = 1.0f;
    return out;
}

// Perspective projection -- Vulkan conventions baked in:
//   Y-down (row 1 negated), depth [0,1], right-handed clip space.

// Reverse-Z perspective (near -> 1, far -> 0). Recommended default.
// Pair with: depth clear = 0.0, compare op = GREATER_OR_EQUAL, D32_SFLOAT.
[[nodiscard]] inline Mat4 perspectiveVk(float fovY, float aspect, float zNear, float zFar) {
    const float f = 1.0f / std::tan(fovY * 0.5f);
    const float inv = 1.0f / (zFar - zNear);

    Mat4 out{};
    out.at(0, 0) = f / aspect;
    out.at(1, 1) = -f;                   // Vulkan Y-flip
    out.at(2, 2) = zNear * inv;          // reverse-Z
    out.at(3, 2) = -1.0f;                // perspective divide
    out.at(2, 3) = (zFar * zNear) * inv; // reverse-Z
    return out;
}

// Infinite far plane + reverse-Z. The "just use this" option.
// Eliminates the "what should far be?" question.
// Pair with: depth clear = 0.0, compare op = GREATER_OR_EQUAL, D32_SFLOAT.
[[nodiscard]] inline Mat4 perspectiveInfiniteReverseZ(float fovY, float aspect, float zNear) {
    const float f = 1.0f / std::tan(fovY * 0.5f);

    Mat4 out{};
    out.at(0, 0) = f / aspect;
    out.at(1, 1) = -f;    // Vulkan Y-flip
    out.at(3, 2) = -1.0f; // perspective divide
    out.at(2, 3) = zNear; // z_ndc = zNear / -z_eye
    return out;
}

// Forward-Z perspective (near -> 0, far -> 1). Legacy / compatibility.
// Pair with: depth clear = 1.0, compare op = LESS_OR_EQUAL, D32_SFLOAT.
[[nodiscard]] inline Mat4 perspectiveForwardZ(float fovY, float aspect, float zNear, float zFar) {
    const float f = 1.0f / std::tan(fovY * 0.5f);
    const float inv = 1.0f / (zNear - zFar);

    Mat4 out{};
    out.at(0, 0) = f / aspect;
    out.at(1, 1) = -f;                   // Vulkan Y-flip
    out.at(2, 2) = zFar * inv;           // forward-Z
    out.at(3, 2) = -1.0f;                // perspective divide
    out.at(2, 3) = (zNear * zFar) * inv; // forward-Z
    return out;
}

// Orthographic projection -- Vulkan conventions (Y-down, depth [0,1]).

// Reverse-Z orthographic (near -> 1, far -> 0).
[[nodiscard]] inline Mat4 orthoVk(float left, float right, float bottom, float top, float zNear,
                                  float zFar) {
    const float rl = right - left;
    const float tb = top - bottom;
    const float fn = zFar - zNear;

    Mat4 out{};
    out.at(0, 0) = 2.0f / rl;
    out.at(1, 1) = -2.0f / tb; // Vulkan Y-flip
    out.at(2, 2) = 1.0f / fn;  // reverse-Z: [1,0]
    out.at(0, 3) = -(right + left) / rl;
    out.at(1, 3) = -(top + bottom) / tb;
    out.at(2, 3) = zFar / fn; // reverse-Z offset
    out.at(3, 3) = 1.0f;
    return out;
}

// View matrix from camera basis vectors (no GLM needed).

// LookAt view matrix. Camera at eye, looking toward target, worldUp = (0,1,0).
// The result is the inverse of the camera's world transform, built directly
// from dot products (no general matrix inverse needed).
[[nodiscard]] inline Mat4 lookAt(const float* eye, const float* target, const float* worldUp) {
    // Forward = normalize(target - eye)
    float fx = target[0] - eye[0];
    float fy = target[1] - eye[1];
    float fz = target[2] - eye[2];
    float fLen = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fLen;
    fy /= fLen;
    fz /= fLen;

    // Right = normalize(cross(worldUp, forward))
    float rx = worldUp[1] * fz - worldUp[2] * fy;
    float ry = worldUp[2] * fx - worldUp[0] * fz;
    float rz = worldUp[0] * fy - worldUp[1] * fx;
    float rLen = std::sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rLen;
    ry /= rLen;
    rz /= rLen;

    // Up = cross(forward, right)
    float ux = fy * rz - fz * ry;
    float uy = fz * rx - fx * rz;
    float uz = fx * ry - fy * rx;

    // View matrix (column-major): rows are right, up, -forward.
    // Translation is -dot(axis, eye).
    Mat4 out{};
    out.at(0, 0) = rx;
    out.at(0, 1) = ry;
    out.at(0, 2) = rz;
    out.at(0, 3) = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);

    out.at(1, 0) = ux;
    out.at(1, 1) = uy;
    out.at(1, 2) = uz;
    out.at(1, 3) = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);

    out.at(2, 0) = -fx;
    out.at(2, 1) = -fy;
    out.at(2, 2) = -fz;
    out.at(2, 3) = (fx * eye[0] + fy * eye[1] + fz * eye[2]);

    out.at(3, 3) = 1.0f;
    return out;
}

// Multiply two 4x4 column-major matrices: result = a * b.
[[nodiscard]] inline Mat4 mat4Mul(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.at(row, k) * b.at(k, col);
            }
            out.at(row, col) = sum;
        }
    }
    return out;
}

} // namespace vksdl
