#include <vksdl/vksdl.hpp>

#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>

// Face normal from three position pointers (CCW winding).
static void faceNormal(float out[4], const float* v0, const float* v1, const float* v2) {
    float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    float nx = e1[1] * e2[2] - e1[2] * e2[1];
    float ny = e1[2] * e2[0] - e1[0] * e2[2];
    float nz = e1[0] * e2[1] - e1[1] * e2[0];
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    out[0] = nx / len;
    out[1] = ny / len;
    out[2] = nz / len;
    out[3] = 0.0f;
}

// Push constants shared between all RT shader stages.
// Must match the GLSL layout exactly (std430 packing).
struct PushConstants {
    float lightDir[3];
    float time;
    float camPos[3];
    float camFov;
    float camRight[3];
    float _pad1;
    float camUp[3];
    float _pad2;
    float camFwd[3];
    float _pad3;
};
static_assert(sizeof(PushConstants) == 80);

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - RT Pyramid [WASD + RMB]", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_rt_pyramid")
                        .requireVulkan(1, 3)
                        .enableWindowSupport()
                        .build()
                        .value();

    auto surface = vksdl::Surface::create(instance, window).value();

    auto device = vksdl::DeviceBuilder(instance, surface)
                      .needSwapchain()
                      .needDynamicRendering()
                      .needSync2()
                      .needRayTracingPipeline()
                      .preferDiscreteGpu()
                      .build()
                      .value();

    std::printf("GPU: %s\n", device.gpuName());

    auto swapchain =
        vksdl::SwapchainBuilder(device, surface).size(window.pixelSize()).build().value();

    auto frames = vksdl::FrameSync::create(device, swapchain.imageCount()).value();
    auto allocator = vksdl::Allocator::create(instance, device).value();

    float pyramidVerts[] = {
        0.0f,  0.7f,  0.0f,  // 0: apex
        0.5f,  -0.3f, 0.5f,  // 1
        -0.5f, -0.3f, 0.5f,  // 2
        -0.5f, -0.3f, -0.5f, // 3
        0.5f,  -0.3f, -0.5f, // 4
    };
    std::uint32_t pyramidIndices[] = {
        0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 3, 2, 1, 4, 3,
    };

    // Compute per-face normals (8 entries: 6 faces + 2 padding).
    struct alignas(16) Vec4 {
        float x, y, z, w;
    };
    Vec4 faceNormals[8];
    for (int i = 0; i < 6; ++i) {
        const float* p0 = &pyramidVerts[pyramidIndices[i * 3 + 0] * 3];
        const float* p1 = &pyramidVerts[pyramidIndices[i * 3 + 1] * 3];
        const float* p2 = &pyramidVerts[pyramidIndices[i * 3 + 2] * 3];
        faceNormal(&faceNormals[i].x, p0, p1, p2);
    }
    faceNormals[6] = {0, 1, 0, 0};
    faceNormals[7] = {0, 1, 0, 0};

    float groundVerts[] = {
        -2.0f, -0.3f, 2.0f, 2.0f, -0.3f, 2.0f, 2.0f, -0.3f, -2.0f, -2.0f, -0.3f, -2.0f,
    };
    std::uint32_t groundIndices[] = {0, 1, 2, 0, 2, 3};

    auto pyramidVBuf = vksdl::BufferBuilder(allocator)
                           .vertexBuffer()
                           .accelerationStructureInput()
                           .size(sizeof(pyramidVerts))
                           .build()
                           .value();
    if (!vksdl::uploadToBuffer(allocator, device, pyramidVBuf, pyramidVerts, sizeof(pyramidVerts))
             .ok())
        return 1;
    auto pyramidIBuf = vksdl::BufferBuilder(allocator)
                           .indexBuffer()
                           .accelerationStructureInput()
                           .size(sizeof(pyramidIndices))
                           .build()
                           .value();
    if (!vksdl::uploadToBuffer(allocator, device, pyramidIBuf, pyramidIndices,
                               sizeof(pyramidIndices))
             .ok())
        return 1;

    auto groundVBuf = vksdl::BufferBuilder(allocator)
                          .vertexBuffer()
                          .accelerationStructureInput()
                          .size(sizeof(groundVerts))
                          .build()
                          .value();
    if (!vksdl::uploadToBuffer(allocator, device, groundVBuf, groundVerts, sizeof(groundVerts))
             .ok())
        return 1;
    auto groundIBuf = vksdl::BufferBuilder(allocator)
                          .indexBuffer()
                          .accelerationStructureInput()
                          .size(sizeof(groundIndices))
                          .build()
                          .value();
    if (!vksdl::uploadToBuffer(allocator, device, groundIBuf, groundIndices, sizeof(groundIndices))
             .ok())
        return 1;

    auto normalBuf =
        vksdl::uploadStorageBuffer(allocator, device, faceNormals, sizeof(faceNormals)).value();

    auto pyramidGeo = vksdl::BlasTriangleGeometry::fromBuffers(pyramidVBuf, pyramidIBuf, 5, 18,
                                                               3 * sizeof(float));
    auto pyramidBlas = vksdl::BlasBuilder(device, allocator)
                           .addTriangles(pyramidGeo)
                           .preferFastTrace()
                           .build()
                           .value();

    auto groundGeo =
        vksdl::BlasTriangleGeometry::fromBuffers(groundVBuf, groundIBuf, 4, 6, 3 * sizeof(float));
    auto groundBlas = vksdl::BlasBuilder(device, allocator)
                          .addTriangles(groundGeo)
                          .preferFastTrace()
                          .build()
                          .value();

    auto identityXform = vksdl::transformIdentity();
    auto tlas = vksdl::TlasBuilder(device, allocator)
                    .addInstance(pyramidBlas, identityXform.matrix, 0, 0xFE)
                    .addInstance(groundBlas, identityXform.matrix, 6)
                    .preferFastTrace()
                    .build()
                    .value();

    auto storageImage = vksdl::ImageBuilder(allocator)
                            .size(swapchain.extent().width, swapchain.extent().height)
                            .format(VK_FORMAT_R8G8B8A8_UNORM)
                            .storage()
                            .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                            .build()
                            .value();

    auto descriptors = vksdl::DescriptorSetBuilder(device)
                           .addAccelerationStructure(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                                            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
                           .addStorageImage(1, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
                           .addStorageBuffer(2, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
                           .build()
                           .value();

    descriptors.updateAccelerationStructure(0, tlas.vkAccelerationStructure());
    descriptors.updateImage(1, storageImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
    descriptors.updateBuffer(2, normalBuf.vkBuffer(), sizeof(faceNormals));

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::RayTracingPipelineBuilder(device)
                        .rayGenShader(shaderDir / "raygen.rgen.spv")
                        .missShader(shaderDir / "miss.rmiss.spv")
                        .missShader(shaderDir / "shadow_miss.rmiss.spv")
                        .closestHitShader(shaderDir / "closesthit.rchit.spv")
                        .descriptorSetLayout(descriptors.vkDescriptorSetLayout())
                        .pushConstants<PushConstants>(vksdl::kAllRtStages)
                        .maxRecursionDepth(3)
                        .build()
                        .value();

    auto sbt = vksdl::ShaderBindingTable::create(device, pipeline, allocator, 2, 1).value();

    vksdl::FlyCamera camera(0.0f, 2.0f, -4.5f, 0.0f, -0.3f);
    camera.setSpeed(3.0f);
    constexpr float fovY = 1.0472f; // 60 deg
    const float camFovTan = std::tan(fovY * 0.5f);

    std::printf("Controls: WASD move, Space/LShift up/down, RMB+drag look, ESC quit\n");

    bool running = true;
    vksdl::Event event;
    auto prevTime = std::chrono::steady_clock::now();
    auto startTime = prevTime;

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prevTime).count();
        float time = std::chrono::duration<float>(now - startTime).count();
        prevTime = now;
        dt = dt > 0.1f ? 0.1f : dt;

        camera.update(dt);
        if (camera.shouldQuit())
            running = false;

        if (window.consumeResize()) {
            (void) swapchain.recreate(device, window);

            storageImage = vksdl::ImageBuilder(allocator)
                               .size(swapchain.extent().width, swapchain.extent().height)
                               .format(VK_FORMAT_R8G8B8A8_UNORM)
                               .storage()
                               .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                               .build()
                               .value();

            descriptors.updateImage(1, storageImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        auto pyramidXform = vksdl::transformRotateY(time * 0.7f);
        tlas = vksdl::TlasBuilder(device, allocator)
                   .addInstance(pyramidBlas, pyramidXform.matrix, 0, 0xFE)
                   .addInstance(groundBlas, identityXform.matrix, 6)
                   .preferFastTrace()
                   .build()
                   .value();

        descriptors.updateAccelerationStructure(0, tlas.vkAccelerationStructure());

        float lightAngle = time * 0.3f;
        constexpr float k = 0.70710678f;

        PushConstants pc{};
        pc.lightDir[0] = std::sin(lightAngle) * k;
        pc.lightDir[1] = k;
        pc.lightDir[2] = std::cos(lightAngle) * k;
        pc.time = time;
        std::memcpy(pc.camPos, camera.position(), 3 * sizeof(float));
        pc.camFov = camFovTan;
        std::memcpy(pc.camRight, camera.right(), 3 * sizeof(float));
        std::memcpy(pc.camUp, camera.up(), 3 * sizeof(float));
        std::memcpy(pc.camFwd, camera.forward(), 3 * sizeof(float));

        vksdl::beginOneTimeCommands(frame.cmd);

        vksdl::transitionToComputeWrite(frame.cmd, storageImage.vkImage());

        pipeline.bind(frame.cmd, descriptors);
        pipeline.pushConstants(frame.cmd, pc);

        sbt.traceRays(frame.cmd, swapchain.extent().width, swapchain.extent().height);

        vksdl::blitToSwapchain(frame.cmd, storageImage, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, img.image, swapchain.extent());

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    }

    device.waitIdle();
    return 0;
}
