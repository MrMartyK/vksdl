#pragma once

#include <vulkan/vulkan.h>

#include <cmath>

namespace vksdl {

// Free functions returning VkTransformMatrixKHR (row-major 3x4).
// No custom type -- fully interoperable with raw Vulkan RT code.

[[nodiscard]] inline VkTransformMatrixKHR transformIdentity() {
    VkTransformMatrixKHR t{};
    t.matrix[0][0] = 1.0f;
    t.matrix[1][1] = 1.0f;
    t.matrix[2][2] = 1.0f;
    return t;
}

[[nodiscard]] inline VkTransformMatrixKHR transformTranslate(float x, float y, float z) {
    VkTransformMatrixKHR t{};
    t.matrix[0][0] = 1.0f;
    t.matrix[0][3] = x;
    t.matrix[1][1] = 1.0f;
    t.matrix[1][3] = y;
    t.matrix[2][2] = 1.0f;
    t.matrix[2][3] = z;
    return t;
}

[[nodiscard]] inline VkTransformMatrixKHR transformScale(float s) {
    VkTransformMatrixKHR t{};
    t.matrix[0][0] = s;
    t.matrix[1][1] = s;
    t.matrix[2][2] = s;
    return t;
}

[[nodiscard]] inline VkTransformMatrixKHR transformTranslateScale(float x, float y, float z,
                                                                  float s) {
    VkTransformMatrixKHR t{};
    t.matrix[0][0] = s;
    t.matrix[0][3] = x;
    t.matrix[1][1] = s;
    t.matrix[1][3] = y;
    t.matrix[2][2] = s;
    t.matrix[2][3] = z;
    return t;
}

[[nodiscard]] inline VkTransformMatrixKHR transformRotateY(float radians) {
    float c = std::cos(radians);
    float sn = std::sin(radians);
    VkTransformMatrixKHR t{};
    t.matrix[0][0] = c;
    t.matrix[0][2] = sn;
    t.matrix[1][1] = 1.0f;
    t.matrix[2][0] = -sn;
    t.matrix[2][2] = c;
    return t;
}

} // namespace vksdl
