#include <vksdl/transform.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>

static constexpr float kEps = 1e-5f;

static bool near(float a, float b) {
    return std::fabs(a - b) < kEps;
}

static void testIdentity() {
    auto t = vksdl::transformIdentity();
    assert(near(t.matrix[0][0], 1.0f));
    assert(near(t.matrix[1][1], 1.0f));
    assert(near(t.matrix[2][2], 1.0f));
    assert(near(t.matrix[0][3], 0.0f));
    assert(near(t.matrix[1][3], 0.0f));
    assert(near(t.matrix[2][3], 0.0f));
}

static void testTranslate() {
    auto t = vksdl::transformTranslate(1.0f, 2.0f, 3.0f);
    assert(near(t.matrix[0][0], 1.0f));
    assert(near(t.matrix[1][1], 1.0f));
    assert(near(t.matrix[2][2], 1.0f));
    assert(near(t.matrix[0][3], 1.0f));
    assert(near(t.matrix[1][3], 2.0f));
    assert(near(t.matrix[2][3], 3.0f));
}

static void testScale() {
    auto t = vksdl::transformScale(5.0f);
    assert(near(t.matrix[0][0], 5.0f));
    assert(near(t.matrix[1][1], 5.0f));
    assert(near(t.matrix[2][2], 5.0f));
    assert(near(t.matrix[0][3], 0.0f));
}

static void testTranslateScale() {
    auto t = vksdl::transformTranslateScale(1.0f, 2.0f, 3.0f, 0.5f);
    assert(near(t.matrix[0][0], 0.5f));
    assert(near(t.matrix[1][1], 0.5f));
    assert(near(t.matrix[2][2], 0.5f));
    assert(near(t.matrix[0][3], 1.0f));
    assert(near(t.matrix[1][3], 2.0f));
    assert(near(t.matrix[2][3], 3.0f));
}

static void testRotateY() {
    // 90 degrees around Y: X -> Z, Z -> -X
    float angle = 3.14159265f * 0.5f;
    auto t = vksdl::transformRotateY(angle);
    // cos(90) ~ 0, sin(90) ~ 1
    assert(near(t.matrix[0][0], 0.0f));
    assert(near(t.matrix[0][2], 1.0f));
    assert(near(t.matrix[1][1], 1.0f));
    assert(near(t.matrix[2][0], -1.0f));
    assert(near(t.matrix[2][2], 0.0f));

    // Identity at 0 degrees
    auto t0 = vksdl::transformRotateY(0.0f);
    assert(near(t0.matrix[0][0], 1.0f));
    assert(near(t0.matrix[2][2], 1.0f));
    assert(near(t0.matrix[0][2], 0.0f));
}

int main() {
    testIdentity();
    testTranslate();
    testScale();
    testTranslateScale();
    testRotateY();

    std::printf("all transform tests passed\n");
    return 0;
}
