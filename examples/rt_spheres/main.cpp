#include <vksdl/vksdl.hpp>

#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

struct Vec3 {
    float x, y, z;
};

static Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
static Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
static Vec3 operator*(Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

static float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static float length(Vec3 v) {
    return std::sqrt(dot(v, v));
}

static Vec3 normalize(Vec3 v) {
    float len = length(v);
    return {v.x / len, v.y / len, v.z / len};
}

// Ray-sphere intersection. Returns distance t, or -1 on miss.
static float raySphereIntersect(Vec3 origin, Vec3 dir, Vec3 center, float radius) {
    Vec3 oc = origin - center;
    float b = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f)
        return -1.0f;
    float sqrtDisc = std::sqrt(disc);
    float t = -b - sqrtDisc;
    if (t < 0.001f)
        t = -b + sqrtDisc;
    return t > 0.001f ? t : -1.0f;
}

struct PushConstants {
    float camPos[3];
    float camFov;
    float camRight[3];
    std::uint32_t frameIdx;
    float camUp[3];
    std::uint32_t sampleCnt;
    float camFwd[3];
    float aperture;
    float focusDist;
    float _pad[3];
};
static_assert(sizeof(PushConstants) == 80);

struct Material {
    float albedo[3];
    std::uint32_t type;  // 0=diffuse, 1=metal, 2=dielectric
    float roughness;     // metal fuzz
    float ior;           // dielectric IOR
    float specRoughness; // GGX roughness for diffuse specular lobe
    float absorption;    // Beer-Lambert scale for dielectrics (0 = clear)
};
static_assert(sizeof(Material) == 32);

struct Sphere {
    Vec3 center;
    float radius;
    Material material;
};

struct MeshData {
    std::vector<float> vertices; // xyz interleaved
    std::vector<std::uint32_t> indices;
};

static MeshData generateIcosphere(int subdivisions) {
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const float a = 1.0f;
    const float b = phi;

    std::vector<Vec3> verts = {
        {-a, b, 0},  {a, b, 0},  {-a, -b, 0}, {a, -b, 0}, {0, -a, b},  {0, a, b},
        {0, -a, -b}, {0, a, -b}, {b, 0, -a},  {b, 0, a},  {-b, 0, -a}, {-b, 0, a},
    };

    for (auto& v : verts)
        v = normalize(v);

    std::vector<std::uint32_t> tris = {
        0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
        4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
        6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1,
    };

    auto midpoint = [&](std::uint32_t i0, std::uint32_t i1) -> std::uint32_t {
        Vec3 mid = normalize((verts[i0] + verts[i1]) * 0.5f);
        verts.push_back(mid);
        return static_cast<std::uint32_t>(verts.size() - 1);
    };

    for (int sub = 0; sub < subdivisions; sub++) {
        std::vector<std::uint32_t> newTris;
        newTris.reserve(tris.size() * 4);

        for (std::size_t i = 0; i < tris.size(); i += 3) {
            std::uint32_t v0 = tris[i], v1 = tris[i + 1], v2 = tris[i + 2];
            std::uint32_t m01 = midpoint(v0, v1);
            std::uint32_t m12 = midpoint(v1, v2);
            std::uint32_t m20 = midpoint(v2, v0);

            newTris.insert(newTris.end(), {v0, m01, m20});
            newTris.insert(newTris.end(), {v1, m12, m01});
            newTris.insert(newTris.end(), {v2, m20, m12});
            newTris.insert(newTris.end(), {m01, m12, m20});
        }
        tris = std::move(newTris);
    }

    MeshData mesh;
    mesh.vertices.reserve(verts.size() * 3);
    for (const auto& v : verts) {
        mesh.vertices.push_back(v.x);
        mesh.vertices.push_back(v.y);
        mesh.vertices.push_back(v.z);
    }
    mesh.indices = std::move(tris);
    return mesh;
}

static std::vector<Sphere> generateScene(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    auto randf = [&]() { return dist01(rng); };
    auto randRange = [&](float lo, float hi) { return lo + (hi - lo) * randf(); };

    std::vector<Sphere> spheres;

    // Ground material entry at index 0.
    spheres.push_back({{0, 0, 0}, 0, {{0.5f, 0.5f, 0.5f}, 0, 0, 0, 0.3f, 0}});

    // 3 large feature spheres.
    spheres.push_back({{0, 1, 0}, 1.0f, {{1.0f, 1.0f, 1.0f}, 2, 0.0f, 1.5f, 0, 0}}); // clear glass
    spheres.push_back({{-4, 1, 0}, 1.0f, {{0.4f, 0.2f, 0.1f}, 0, 0.0f, 0.0f, 0.5f, 0}}); // diffuse
    spheres.push_back({{4, 1, 0}, 1.0f, {{0.7f, 0.6f, 0.5f}, 1, 0.0f, 0.0f, 0, 0}});     // metal

    // Small random spheres.
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            Vec3 center = {a + 0.9f * randf(), 0.2f, b + 0.9f * randf()};

            if (length(center - Vec3{0, 0.2f, 0}) < 1.2f)
                continue;
            if (length(center - Vec3{-4, 0.2f, 0}) < 1.2f)
                continue;
            if (length(center - Vec3{4, 0.2f, 0}) < 1.2f)
                continue;

            float chooseMat = randf();
            Material mat{};

            if (chooseMat < 0.65f) {
                // Diffuse with GGX specular lobe.
                mat.albedo[0] = randf() * randf();
                mat.albedo[1] = randf() * randf();
                mat.albedo[2] = randf() * randf();
                mat.type = 0;
                mat.specRoughness = randRange(0.3f, 0.8f);
            } else if (chooseMat < 0.85f) {
                // Metal.
                mat.albedo[0] = randRange(0.5f, 1.0f);
                mat.albedo[1] = randRange(0.5f, 1.0f);
                mat.albedo[2] = randRange(0.5f, 1.0f);
                mat.type = 1;
                mat.roughness = randRange(0.0f, 0.3f);
            } else {
                // Dielectric (glass) -- some clear, some tinted.
                mat.type = 2;
                mat.ior = 1.5f;
                if (randf() < 0.4f) {
                    // Tinted glass: warm/cool random tint.
                    mat.albedo[0] = randRange(0.5f, 1.0f);
                    mat.albedo[1] = randRange(0.3f, 0.9f);
                    mat.albedo[2] = randRange(0.2f, 0.8f);
                    mat.absorption = randRange(1.0f, 4.0f);
                } else {
                    // Clear glass.
                    mat.albedo[0] = 1.0f;
                    mat.albedo[1] = 1.0f;
                    mat.albedo[2] = 1.0f;
                    mat.absorption = 0.0f;
                }
            }

            spheres.push_back({center, 0.2f, mat});
        }
    }

    return spheres;
}

int main() {
    auto app = vksdl::App::create().value();
    auto window = app.createWindow("vksdl - RT Spheres [orbit]", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
                        .appName("vksdl_rt_spheres")
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

    std::mt19937 rng(42);
    auto spheres = generateScene(rng);
    std::printf("Scene: %zu spheres\n", spheres.size());

    auto sphereMesh = generateIcosphere(3);
    std::printf("Icosphere: %zu vertices, %zu triangles\n", sphereMesh.vertices.size() / 3,
                sphereMesh.indices.size() / 3);

    VkDeviceSize sphereVSize = sphereMesh.vertices.size() * sizeof(float);
    auto sphereVBuf = vksdl::BufferBuilder(allocator)
                          .vertexBuffer()
                          .accelerationStructureInput()
                          .size(sphereVSize)
                          .build()
                          .value();
    if (!vksdl::uploadToBuffer(allocator, device, sphereVBuf, sphereMesh.vertices.data(),
                               sphereVSize)
             .ok())
        return 1;
    VkDeviceSize sphereISize = sphereMesh.indices.size() * sizeof(std::uint32_t);
    auto sphereIBuf = vksdl::BufferBuilder(allocator)
                          .indexBuffer()
                          .accelerationStructureInput()
                          .size(sphereISize)
                          .build()
                          .value();
    if (!vksdl::uploadToBuffer(allocator, device, sphereIBuf, sphereMesh.indices.data(),
                               sphereISize)
             .ok())
        return 1;

    float groundVerts[] = {
        -50.0f, 0.0f, 50.0f, 50.0f, 0.0f, 50.0f, 50.0f, 0.0f, -50.0f, -50.0f, 0.0f, -50.0f,
    };
    std::uint32_t groundIndices[] = {0, 1, 2, 0, 2, 3};

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

    std::vector<Material> materials;
    materials.reserve(spheres.size());
    for (const auto& s : spheres)
        materials.push_back(s.material);

    auto materialBuf = vksdl::uploadStorageBuffer(allocator, device, materials.data(),
                                                  materials.size() * sizeof(Material))
                           .value();

    auto sphereGeo = vksdl::BlasTriangleGeometry::fromBuffers(
        sphereVBuf, sphereIBuf, static_cast<std::uint32_t>(sphereMesh.vertices.size() / 3),
        static_cast<std::uint32_t>(sphereMesh.indices.size()), 3 * sizeof(float));

    auto sphereBlas = vksdl::BlasBuilder(device, allocator)
                          .addTriangles(sphereGeo)
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

    auto buildTlas = [&]() {
        vksdl::TlasBuilder builder(device, allocator);

        auto groundXform = vksdl::transformIdentity();
        builder.addInstance(groundBlas, groundXform.matrix, 0);

        for (std::size_t i = 1; i < spheres.size(); i++) {
            auto xform = vksdl::transformTranslateScale(spheres[i].center.x, spheres[i].center.y,
                                                        spheres[i].center.z, spheres[i].radius);
            builder.addInstance(sphereBlas, xform.matrix, static_cast<std::uint32_t>(i));
        }

        return builder.preferFastTrace().build().value();
    };

    auto tlas = buildTlas();

    auto createImages = [&](std::uint32_t w, std::uint32_t h) {
        auto accum = vksdl::ImageBuilder(allocator)
                         .size(w, h)
                         .format(VK_FORMAT_R32G32B32A32_SFLOAT)
                         .storage()
                         .addUsage(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                         .build()
                         .value();

        auto display = vksdl::ImageBuilder(allocator)
                           .size(w, h)
                           .format(VK_FORMAT_R8G8B8A8_UNORM)
                           .storage()
                           .addUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                           .build()
                           .value();

        return std::make_pair(std::move(accum), std::move(display));
    };

    auto [accumImage, displayImage] =
        createImages(swapchain.extent().width, swapchain.extent().height);

    auto descriptors = vksdl::DescriptorSetBuilder(device)
                           .addAccelerationStructure(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                                            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
                           .addStorageImage(1, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
                           .addStorageImage(2, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
                           .addStorageBuffer(3, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
                           .build()
                           .value();

    auto updateDescriptors = [&]() {
        descriptors.updateAccelerationStructure(0, tlas.vkAccelerationStructure());
        descriptors.updateImage(1, accumImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        descriptors.updateImage(2, displayImage.vkImageView(), VK_IMAGE_LAYOUT_GENERAL);
        descriptors.updateBuffer(3, materialBuf.vkBuffer(), materials.size() * sizeof(Material));
    };
    updateDescriptors();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";

    auto pipeline = vksdl::RayTracingPipelineBuilder(device)
                        .rayGenShader(shaderDir / "raygen.rgen.spv")
                        .missShader(shaderDir / "miss.rmiss.spv")
                        .closestHitShader(shaderDir / "closesthit.rchit.spv")
                        .descriptorSetLayout(descriptors.vkDescriptorSetLayout())
                        .pushConstants<PushConstants>(vksdl::kAllRtStages)
                        .maxRecursionDepth(2)
                        .build()
                        .value();

    auto sbt = vksdl::ShaderBindingTable::create(device, pipeline, allocator, 1, 1).value();

    // Orbit camera centered on the glass sphere, classic RTOW viewpoint.
    // FOV stored in radians on the camera; push constant needs tan(fovY/2).

    constexpr float fovY = 0.7f; // ~40 deg -- narrow for classic RTOW look
    vksdl::OrbitCamera camera(0.0f, 1.0f, 0.0f, 13.0f, 0.2f, -0.15f);
    camera.setFovY(fovY);
    camera.setViewport(swapchain.extent().width, swapchain.extent().height);
    camera.setDistanceLimits(1.0f, 100.0f);

    const float camFovTan = std::tan(fovY * 0.5f);

    std::uint32_t frameIdx = 0;
    std::uint32_t sampleCnt = 0;
    bool accumDirty = true;

    std::printf("Controls: LMB orbit, RMB/MMB pan, scroll zoom, dblclick focus, ESC quit\n");

    bool running = true;
    vksdl::Event event;
    auto prevTime = std::chrono::steady_clock::now();

    while (running) {
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
            if (event.type == vksdl::EventType::MouseWheel) {
                camera.feedScrollDelta(event.scroll);
            }
            // Double-click: pick a sphere to focus on.
            if (event.type == vksdl::EventType::MouseButtonDown && event.button == 1 &&
                event.clicks >= 2) {
                float px = event.mouseX;
                float py = event.mouseY;
                float w = static_cast<float>(swapchain.extent().width);
                float h = static_cast<float>(swapchain.extent().height);
                float aspect = w / h;

                // Screen coords to [-1,1], Y flipped.
                float u = (px / w) * 2.0f - 1.0f;
                float v = -((py / h) * 2.0f - 1.0f);

                // Ray from camera through pixel (same math as raygen.rgen).
                Vec3 origin = {camera.position()[0], camera.position()[1], camera.position()[2]};
                Vec3 fwd = {camera.forward()[0], camera.forward()[1], camera.forward()[2]};
                Vec3 right = {camera.right()[0], camera.right()[1], camera.right()[2]};
                Vec3 up = {camera.up()[0], camera.up()[1], camera.up()[2]};
                Vec3 dir = normalize(fwd + right * (u * aspect * camFovTan) + up * (v * camFovTan));

                // Test all spheres, find nearest hit.
                float bestT = 1e30f;
                int bestIdx = -1;
                for (std::size_t i = 1; i < spheres.size(); i++) {
                    float t = raySphereIntersect(origin, dir, spheres[i].center, spheres[i].radius);
                    if (t > 0.0f && t < bestT) {
                        bestT = t;
                        bestIdx = static_cast<int>(i);
                    }
                }
                // Also test ground (large sphere at y=-1000, radius 1000).
                // Skip -- ground isn't interesting to focus on.

                if (bestIdx >= 0) {
                    const auto& s = spheres[bestIdx];
                    camera.setTarget(s.center.x, s.center.y, s.center.z);
                    // Distance = from eye to sphere center.
                    float d = length(origin - s.center);
                    camera.setDistance(d);
                    accumDirty = true;
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - prevTime).count();
        prevTime = now;
        dt = dt > 0.1f ? 0.1f : dt;

        if (camera.update(dt))
            accumDirty = true;
        if (camera.shouldQuit())
            running = false;

        if (window.consumeResize()) {
            (void) swapchain.recreate(device, window);

            auto [newAccum, newDisplay] =
                createImages(swapchain.extent().width, swapchain.extent().height);
            accumImage = std::move(newAccum);
            displayImage = std::move(newDisplay);
            updateDescriptors();
            camera.setViewport(swapchain.extent().width, swapchain.extent().height);
            accumDirty = true;
        }

        if (accumDirty) {
            sampleCnt = 0;
            frameIdx = 0;
            accumDirty = false;
        }

        sampleCnt++;
        frameIdx++;

        if ((sampleCnt & 15) == 0) {
            char title[128];
            std::snprintf(title, sizeof(title), "vksdl - RT Spheres (%u samples)", sampleCnt);
            SDL_SetWindowTitle(window.sdlWindow(), title);
        }

        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        PushConstants pc{};
        std::memcpy(pc.camPos, camera.position(), 3 * sizeof(float));
        pc.camFov = camFovTan;
        std::memcpy(pc.camRight, camera.right(), 3 * sizeof(float));
        pc.frameIdx = frameIdx;
        std::memcpy(pc.camUp, camera.up(), 3 * sizeof(float));
        pc.sampleCnt = sampleCnt;
        std::memcpy(pc.camFwd, camera.forward(), 3 * sizeof(float));
        pc.aperture = 0.08f;
        pc.focusDist = camera.distance();

        vksdl::beginOneTimeCommands(frame.cmd);

        if (sampleCnt == 1) {
            VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
            vksdl::clearImage(frame.cmd, accumImage, clearColor, VK_IMAGE_LAYOUT_GENERAL);
        }

        if (sampleCnt > 1) {
            VkMemoryBarrier2 memBarrier{};
            memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            memBarrier.dstAccessMask =
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &memBarrier;
            vkCmdPipelineBarrier2(frame.cmd, &dep);
        }

        vksdl::transitionToComputeWrite(frame.cmd, displayImage.vkImage());

        pipeline.bind(frame.cmd, descriptors);
        pipeline.pushConstants(frame.cmd, pc);

        sbt.traceRays(frame.cmd, swapchain.extent().width, swapchain.extent().height);

        vksdl::blitToSwapchain(frame.cmd, displayImage, VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, img.image, swapchain.extent());

        vksdl::endCommands(frame.cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    }

    device.waitIdle();
    return 0;
}
