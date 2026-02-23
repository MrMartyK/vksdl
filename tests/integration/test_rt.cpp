#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

// Helper: check if the GPU supports RT by attempting to create a device with
// needRayTracingPipeline(). Returns false when the GPU lacks the extensions.
static bool gpuSupportsRt(const vksdl::Instance& instance,
                           const vksdl::Surface& surface) {
    auto result = vksdl::DeviceBuilder(instance, surface)
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .needRayTracingPipeline()
        .preferDiscreteGpu()
        .build();
    return result.ok();
}

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("rt test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
        .appName("test_rt")
        .requireVulkan(1, 3)
        .validation(vksdl::Validation::Off)
        .enableWindowSupport()
        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    if (!gpuSupportsRt(instance.value(), surface.value())) {
        std::printf("GPU does not support ray tracing -- skipping RT tests\n");
        std::printf("rt test passed (skipped)\n");
        return 0;
    }

    auto device = vksdl::DeviceBuilder(instance.value(), surface.value())
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .needRayTracingPipeline()
        .preferDiscreteGpu()
        .build();
    assert(device.ok());

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device.value().vkPhysicalDevice(), &props);
    std::printf("  GPU: %s\n", props.deviceName);

    {
        assert(device.value().shaderGroupHandleSize() > 0);
        assert(device.value().shaderGroupBaseAlignment() > 0);
        assert(device.value().shaderGroupHandleAlignment() > 0);
        assert(device.value().maxRayRecursionDepth() > 0);
        assert(device.value().minAccelerationStructureScratchOffsetAlignment() > 0);

        std::printf("  RT properties (handleSize=%u, baseAlign=%u, maxRecursion=%u): ok\n",
                    device.value().shaderGroupHandleSize(),
                    device.value().shaderGroupBaseAlignment(),
                    device.value().maxRayRecursionDepth());
    }

    {
        auto dev2 = vksdl::DeviceBuilder(instance.value(), surface.value())
            .needSwapchain()
            .requireFeatures12([](VkPhysicalDeviceVulkan12Features& f) {
                f.bufferDeviceAddress = VK_TRUE;
            })
            .build();
        assert(dev2.ok());
        std::printf("  requireFeatures12: ok\n");
    }

    auto allocator = vksdl::Allocator::create(instance.value(), device.value());
    assert(allocator.ok());

    {
        float verts[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        auto buf = vksdl::BufferBuilder(allocator.value())
            .size(sizeof(verts))
            .vertexBuffer()
            .build();
        assert(buf.ok());
        assert(buf.value().deviceAddress() != 0);
        std::printf("  Buffer::deviceAddress(): ok\n");
    }

    vksdl::Blas blas = [&]() {
        float verts[] = {
            -0.5f, -0.5f, 0.0f,
             0.5f, -0.5f, 0.0f,
             0.0f,  0.5f, 0.0f,
        };
        std::uint32_t indices[] = {0, 1, 2};

        auto vbuf = vksdl::BufferBuilder(allocator.value())
            .size(sizeof(verts))
            .vertexBuffer()
            .build();
        assert(vbuf.ok());
        (void)vksdl::uploadToBuffer(allocator.value(), device.value(),
                                    vbuf.value(), verts, sizeof(verts));

        auto ibuf = vksdl::BufferBuilder(allocator.value())
            .size(sizeof(indices))
            .indexBuffer()
            .build();
        assert(ibuf.ok());
        (void)vksdl::uploadToBuffer(allocator.value(), device.value(),
                                    ibuf.value(), indices, sizeof(indices));

        vksdl::BlasTriangleGeometry geo{};
        geo.vertexBufferAddress = vbuf.value().deviceAddress();
        geo.indexBufferAddress  = ibuf.value().deviceAddress();
        geo.vertexCount  = 3;
        geo.indexCount   = 3;
        geo.vertexStride = 3 * sizeof(float);
        geo.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geo.indexType    = VK_INDEX_TYPE_UINT32;
        geo.opaque       = true;

        auto result = vksdl::BlasBuilder(device.value(), allocator.value())
            .addTriangles(geo)
            .preferFastTrace()
            .build();
        assert(result.ok());
        assert(result.value().vkAccelerationStructure() != VK_NULL_HANDLE);
        assert(result.value().deviceAddress() != 0);
        std::printf("  BLAS creation: ok\n");
        return std::move(result.value());
    }();

    {
        auto moved = std::move(blas);
        assert(moved.vkAccelerationStructure() != VK_NULL_HANDLE);
        assert(moved.deviceAddress() != 0);
        blas = std::move(moved);
        std::printf("  BLAS move semantics: ok\n");
    }

    vksdl::Tlas tlas = [&]() {
        float identity[3][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0}};

        auto result = vksdl::TlasBuilder(device.value(), allocator.value())
            .addInstance(blas, identity)
            .preferFastTrace()
            .build();
        assert(result.ok());
        assert(result.value().vkAccelerationStructure() != VK_NULL_HANDLE);
        std::printf("  TLAS creation: ok\n");
        return std::move(result.value());
    }();

    {
        auto moved = std::move(tlas);
        assert(moved.vkAccelerationStructure() != VK_NULL_HANDLE);
        tlas = std::move(moved);
        std::printf("  TLAS move semantics: ok\n");
    }

    auto descriptors = vksdl::DescriptorSetBuilder(device.value())
        .addAccelerationStructure(0, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        .addStorageImage(1, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        .build();
    assert(descriptors.ok());

    descriptors.value().updateAccelerationStructure(0, tlas.vkAccelerationStructure());
    std::printf("  DescriptorSet AS binding: ok\n");

    std::filesystem::path shaderDir =
        std::filesystem::path(SDL_GetBasePath()) / "shaders";

    auto pipeline = vksdl::RayTracingPipelineBuilder(device.value())
        .rayGenShader(shaderDir / "raygen.rgen.spv")
        .missShader(shaderDir / "miss.rmiss.spv")
        .closestHitShader(shaderDir / "closesthit.rchit.spv")
        .descriptorSetLayout(descriptors.value().vkDescriptorSetLayout())
        .maxRecursionDepth(1)
        .build();
    assert(pipeline.ok());
    assert(pipeline.value().vkPipeline() != VK_NULL_HANDLE);
    assert(pipeline.value().vkPipelineLayout() != VK_NULL_HANDLE);
    std::printf("  RT pipeline creation: ok\n");

    {
        auto sbt = vksdl::ShaderBindingTable::create(
            device.value(), pipeline.value(), allocator.value(), 1, 1);
        assert(sbt.ok());

        assert(sbt.value().raygenRegion().deviceAddress != 0);
        assert(sbt.value().raygenRegion().stride > 0);
        assert(sbt.value().raygenRegion().size > 0);
        assert(sbt.value().missRegion().deviceAddress != 0);
        assert(sbt.value().missRegion().stride > 0);
        assert(sbt.value().hitRegion().deviceAddress != 0);
        assert(sbt.value().hitRegion().stride > 0);
        std::printf("  SBT creation: ok\n");

        auto moved = std::move(sbt.value());
        assert(moved.raygenRegion().deviceAddress != 0);
        auto moved2 = std::move(moved);
        assert(moved2.raygenRegion().deviceAddress != 0);
        std::printf("  SBT move semantics: ok\n");
    }

    {
        float verts[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
        std::uint32_t indices[] = {0, 1, 2};

        auto vbuf = vksdl::BufferBuilder(allocator.value())
            .size(sizeof(verts)).vertexBuffer().build();
        assert(vbuf.ok());

        vksdl::BlasTriangleGeometry geo{};
        geo.vertexBufferAddress = vbuf.value().deviceAddress();
        geo.indexBufferAddress  = 0;
        geo.vertexCount  = 3;
        geo.indexCount   = 0;
        geo.vertexStride = 3 * sizeof(float);
        geo.opaque       = true;

        auto builder = vksdl::BlasBuilder(device.value(), allocator.value())
            .addTriangles(geo);
        auto sizes = builder.sizes();
        assert(sizes.ok());
        assert(sizes.value().accelerationStructureSize > 0);
        assert(sizes.value().buildScratchSize > 0);
        std::printf("  BlasBuildSizes query: ok\n");
    }

    device.value().waitIdle();
    std::printf("rt test passed\n");
    return 0;
}
