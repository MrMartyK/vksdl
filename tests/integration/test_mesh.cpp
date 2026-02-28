#include <vksdl/vksdl.hpp>
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

int main() {
    auto app = vksdl::App::create();
    assert(app.ok());

    auto window = app.value().createWindow("mesh test", 640, 480);
    assert(window.ok());

    auto instance = vksdl::InstanceBuilder{}
                        .appName("test_mesh")
                        .requireVulkan(1, 3)
                        .validation(vksdl::Validation::Off)
                        .enableWindowSupport()
                        .build();
    assert(instance.ok());

    auto surface = vksdl::Surface::create(instance.value(), window.value());
    assert(surface.ok());

    auto device = vksdl::DeviceBuilder(instance.value(), surface.value())
                      .needSwapchain()
                      .needDynamicRendering()
                      .needSync2()
                      .preferDiscreteGpu()
                      .build();
    assert(device.ok());

    auto allocator = vksdl::Allocator::create(instance.value(), device.value());
    assert(allocator.ok());

    std::filesystem::path basePath = SDL_GetBasePath();
    std::filesystem::path assetDir = basePath / "assets";

    // 1. loadModel glTF -- load Box.glb, verify structure
    {
        auto model = vksdl::loadModel(assetDir / "Box.glb");
        assert(model.ok());
        assert(!model.value().meshes.empty());

        const auto& mesh = model.value().meshes[0];
        assert(!mesh.vertices.empty());
        assert(!mesh.indices.empty());

        // Check positions are within reasonable bounds for a unit box
        for (const auto& v : mesh.vertices) {
            assert(v.position[0] >= -2.0f && v.position[0] <= 2.0f);
            assert(v.position[1] >= -2.0f && v.position[1] <= 2.0f);
            assert(v.position[2] >= -2.0f && v.position[2] <= 2.0f);
        }

        std::printf("  loadModel glTF (%zu meshes, %zu verts): ok\n", model.value().meshes.size(),
                    mesh.vertices.size());
    }

    // 2. loadModel OBJ -- write minimal OBJ to temp file, load it
    {
        std::filesystem::path tmpObj = basePath / "test_triangle.obj";
        {
            std::ofstream out(tmpObj);
            out << "v 0 0 0\n";
            out << "v 1 0 0\n";
            out << "v 0 1 0\n";
            out << "f 1 2 3\n";
        }

        auto model = vksdl::loadModel(tmpObj);
        assert(model.ok());
        assert(!model.value().meshes.empty());

        const auto& mesh = model.value().meshes[0];
        assert(mesh.vertices.size() == 3);
        assert(mesh.indices.size() == 3);

        // Verify positions
        assert(mesh.vertices[0].position[0] == 0.0f);
        assert(mesh.vertices[1].position[0] == 1.0f);
        assert(mesh.vertices[2].position[1] == 1.0f);

        // Normals should be generated (flat) since OBJ has none
        // For a CCW triangle in the XY plane, normal should point in +Z or -Z
        float nz = mesh.vertices[0].normal[2];
        assert(nz != 0.0f);

        std::filesystem::remove(tmpObj);
        std::printf("  loadModel OBJ (3 verts, flat normals): ok\n");
    }

    // 3. loadModel unknown extension -- returns error
    {
        auto model = vksdl::loadModel("foo.xyz");
        assert(!model.ok());
        std::printf("  loadModel unknown extension: ok\n");
    }

    // 4. loadModel missing file -- returns error
    {
        auto model = vksdl::loadModel("nonexistent.glb");
        assert(!model.ok());
        std::printf("  loadModel missing file: ok\n");
    }

    // 5. MeshData size helpers
    {
        auto model = vksdl::loadModel(assetDir / "Box.glb");
        assert(model.ok());
        const auto& mesh = model.value().meshes[0];

        VkDeviceSize expectedVertex =
            static_cast<VkDeviceSize>(mesh.vertices.size()) * sizeof(vksdl::Vertex);
        VkDeviceSize expectedIndex =
            static_cast<VkDeviceSize>(mesh.indices.size()) * sizeof(std::uint32_t);

        assert(mesh.vertexSizeBytes() == expectedVertex);
        assert(mesh.indexSizeBytes() == expectedIndex);
        assert(mesh.vertexSizeBytes() > 0);
        assert(mesh.indexSizeBytes() > 0);

        std::printf("  MeshData size helpers: ok\n");
    }

    // 6. uploadMesh -- verify GPU handles
    {
        auto model = vksdl::loadModel(assetDir / "Box.glb");
        assert(model.ok());

        auto mesh = vksdl::uploadMesh(allocator.value(), device.value(), model.value().meshes[0]);
        assert(mesh.ok());
        assert(mesh.value().vkVertexBuffer() != VK_NULL_HANDLE);
        assert(mesh.value().vkIndexBuffer() != VK_NULL_HANDLE);
        assert(mesh.value().indexCount() > 0);
        assert(mesh.value().vertexCount() > 0);

        std::printf("  uploadMesh (%u verts, %u indices): ok\n", mesh.value().vertexCount(),
                    mesh.value().indexCount());
    }

    // 7. Mesh move -- verify handles transferred
    {
        auto model = vksdl::loadModel(assetDir / "Box.glb");
        assert(model.ok());

        auto mesh = vksdl::uploadMesh(allocator.value(), device.value(), model.value().meshes[0]);
        assert(mesh.ok());

        VkBuffer vertHandle = mesh.value().vkVertexBuffer();
        VkBuffer idxHandle = mesh.value().vkIndexBuffer();
        std::uint32_t idxCount = mesh.value().indexCount();

        vksdl::Mesh moved = std::move(mesh.value());
        assert(moved.vkVertexBuffer() == vertHandle);
        assert(moved.vkIndexBuffer() == idxHandle);
        assert(moved.indexCount() == idxCount);

        std::printf("  Mesh move: ok\n");
    }

    // 8. ModelData move -- verify meshes transferred
    {
        auto model = vksdl::loadModel(assetDir / "Box.glb");
        assert(model.ok());

        std::size_t meshCount = model.value().meshes.size();
        assert(meshCount > 0);

        vksdl::ModelData moved = std::move(model.value());
        assert(moved.meshes.size() == meshCount);
        assert(!moved.meshes[0].vertices.empty());

        std::printf("  ModelData move: ok\n");
    }

    // 9. Multi-mesh model -- verify multiple meshes load
    // Box.glb may be single-mesh, so we test the OBJ path with multiple shapes
    {
        std::filesystem::path tmpObj = basePath / "test_multi.obj";
        {
            std::ofstream out(tmpObj);
            out << "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
            out << "v 2 0 0\nv 3 0 0\nv 2 1 0\n";
            out << "o shape1\nf 1 2 3\n";
            out << "o shape2\nf 4 5 6\n";
        }

        auto model = vksdl::loadModel(tmpObj);
        assert(model.ok());
        assert(model.value().meshes.size() >= 2);

        std::printf("  multi-mesh OBJ (%zu meshes): ok\n", model.value().meshes.size());

        std::filesystem::remove(tmpObj);
    }

    std::printf("all mesh tests passed\n");
    return 0;
}
