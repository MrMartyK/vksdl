#include <vksdl/pipeline.hpp>
#include <vksdl/shader_reflect.hpp>
#include <vksdl/util.hpp>

#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

static std::filesystem::path shaderDir() {
    return vksdl::exeDir() / "shaders";
}

static void testReflectTriangleShaders() {
    std::printf("reflectSpvFile: triangle vertex shader\n");

    auto vertPath = shaderDir() / "triangle.vert.spv";
    auto fragPath = shaderDir() / "triangle.frag.spv";

    auto vertRefl = vksdl::reflectSpvFile(vertPath, VK_SHADER_STAGE_VERTEX_BIT);
    assert(vertRefl.ok());
    assert(vertRefl.value().bindings.empty());
    assert(vertRefl.value().pushConstants.empty());

    auto fragRefl = vksdl::reflectSpvFile(fragPath, VK_SHADER_STAGE_FRAGMENT_BIT);
    assert(fragRefl.ok());
    assert(fragRefl.value().bindings.empty());
    assert(fragRefl.value().pushConstants.empty());

    auto merged = vksdl::mergeReflections(vertRefl.value(), fragRefl.value());
    assert(merged.ok());
    assert(merged.value().bindings.empty());
    assert(merged.value().pushConstants.empty());
}

static void testReflectComputeShader() {
    std::printf("reflectSpvFile: compute shader with storage image + push constants\n");

    auto compPath = shaderDir() / "reflect_test.comp.spv";
    auto refl = vksdl::reflectSpvFile(compPath, VK_SHADER_STAGE_COMPUTE_BIT);
    assert(refl.ok());

    const auto& layout = refl.value();
    assert(layout.bindings.size() == 1);
    assert(layout.bindings[0].set == 0);
    assert(layout.bindings[0].binding == 0);
    assert(layout.bindings[0].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    assert(layout.bindings[0].stages == VK_SHADER_STAGE_COMPUTE_BIT);

    assert(layout.pushConstants.size() == 1);
    assert(layout.pushConstants[0].stageFlags == VK_SHADER_STAGE_COMPUTE_BIT);
    assert(layout.pushConstants[0].size == 8);
}

static void testReflectFromCode() {
    std::printf("reflectSpv: from in-memory SPIR-V code\n");

    auto compPath = shaderDir() / "reflect_test.comp.spv";
    auto code = vksdl::readSpv(compPath);
    assert(code.ok());

    auto refl = vksdl::reflectSpv(code.value(), VK_SHADER_STAGE_COMPUTE_BIT);
    assert(refl.ok());
    assert(refl.value().bindings.size() == 1);
}

static void testMergeConflict() {
    std::printf("mergeReflections: type conflict\n");

    vksdl::ReflectedLayout a;
    a.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT});

    vksdl::ReflectedLayout b;
    b.bindings.push_back(
        {0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT});

    auto merged = vksdl::mergeReflections(a, b);
    assert(!merged.ok());
}

static void testMergeStageCombine() {
    std::printf("mergeReflections: stage flag combination\n");

    vksdl::ReflectedLayout a;
    a.bindings.push_back({0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT});

    vksdl::ReflectedLayout b;
    b.bindings.push_back(
        {0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT});

    auto merged = vksdl::mergeReflections(a, b);
    assert(merged.ok());
    assert(merged.value().bindings.size() == 1);

    auto expected = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    assert(merged.value().bindings[0].stages == expected);
}

int main() {
    testReflectTriangleShaders();
    testReflectComputeShader();
    testReflectFromCode();
    testMergeConflict();
    testMergeStageCombine();

    std::printf("all shader reflection tests passed\n");
    return 0;
}
