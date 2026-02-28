#include <vksdl/projection.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>

static constexpr float kEps = 1e-5f;

static bool near(float a, float b) {
    return std::fabs(a - b) < kEps;
}

static void testIdentity() {
    auto m = vksdl::mat4Identity();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            assert(near(m.at(i, j), (i == j) ? 1.0f : 0.0f));
}

static void testMat4Mul() {
    auto I = vksdl::mat4Identity();
    auto P = vksdl::perspectiveForwardZ(1.0f, 1.0f, 0.1f, 100.0f);
    auto IP = vksdl::mat4Mul(I, P);
    for (int i = 0; i < 16; ++i)
        assert(near(IP[i], P[i]));
}

static void testPerspectiveForwardZ() {
    auto P = vksdl::perspectiveForwardZ(1.0f, 1.0f, 0.1f, 100.0f);
    // Y is flipped for Vulkan
    assert(P.at(1, 1) < 0.0f);
    // Perspective divide column
    assert(near(P.at(3, 2), -1.0f));
    // W row is zero except for perspective divide
    assert(near(P.at(3, 0), 0.0f));
    assert(near(P.at(3, 1), 0.0f));
    assert(near(P.at(3, 3), 0.0f));
}

static void testPerspectiveReverseZ() {
    float zNear = 0.1f, zFar = 100.0f;
    auto P = vksdl::perspectiveVk(1.0f, 1.0f, zNear, zFar);
    assert(P.at(1, 1) < 0.0f);
    assert(near(P.at(3, 2), -1.0f));

    // Reverse-Z: near plane maps to NDC depth 1, far plane maps to 0.
    // clip_z = P[2][2]*z + P[2][3], clip_w = P[3][2]*z = -z
    // ndc_z  = clip_z / clip_w
    auto ndcDepth = [&](float viewZ) {
        float clip_z = P.at(2, 2) * viewZ + P.at(2, 3);
        float clip_w = P.at(3, 2) * viewZ;
        return clip_z / clip_w;
    };
    assert(near(ndcDepth(-zNear), 1.0f)); // near -> 1
    assert(near(ndcDepth(-zFar), 0.0f));  // far  -> 0
}

static void testPerspectiveInfiniteReverseZ() {
    auto P = vksdl::perspectiveInfiniteReverseZ(1.0f, 1.0f, 0.1f);
    assert(P.at(1, 1) < 0.0f);
    assert(near(P.at(3, 2), -1.0f));
    assert(near(P.at(2, 3), 0.1f));
}

static void testOrtho() {
    auto O = vksdl::orthoVk(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    // X scale = 2 / (right - left) = 1
    assert(near(O.at(0, 0), 1.0f));
    // Y flipped: -2 / (top - bottom) = -1
    assert(near(O.at(1, 1), -1.0f));
    // W is 1 (orthographic, no perspective divide)
    assert(near(O.at(3, 3), 1.0f));
    assert(near(O.at(3, 2), 0.0f));
}

static void testLookAt() {
    float eye[] = {0.0f, 0.0f, 5.0f};
    float target[] = {0.0f, 0.0f, 0.0f};
    float up[] = {0.0f, 1.0f, 0.0f};
    auto V = vksdl::lookAt(eye, target, up);

    // Camera at origin should produce identity-like translation
    // The view matrix should transform eye position to origin
    // V * eye = [0, 0, 0, 1] (in homogeneous coords)
    float ex = V.at(0, 0) * eye[0] + V.at(0, 1) * eye[1] + V.at(0, 2) * eye[2] + V.at(0, 3);
    float ey = V.at(1, 0) * eye[0] + V.at(1, 1) * eye[1] + V.at(1, 2) * eye[2] + V.at(1, 3);
    float ez = V.at(2, 0) * eye[0] + V.at(2, 1) * eye[1] + V.at(2, 2) * eye[2] + V.at(2, 3);
    assert(near(ex, 0.0f));
    assert(near(ey, 0.0f));
    assert(near(ez, 0.0f));
}

static void testAt() {
    vksdl::Mat4 m{};
    m.at(2, 3) = 42.0f;
    // Column-major: col 3, row 2 -> index 3*4+2 = 14
    assert(near(m[14], 42.0f));
    assert(near(m.at(2, 3), 42.0f));
}

int main() {
    testIdentity();
    testAt();
    testMat4Mul();
    testPerspectiveForwardZ();
    testPerspectiveReverseZ();
    testPerspectiveInfiniteReverseZ();
    testOrtho();
    testLookAt();

    std::printf("all projection tests passed\n");
    return 0;
}
