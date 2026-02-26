#pragma once

#include <vksdl/error.hpp>
#include <vksdl/result.hpp>
#include <vksdl/vma_fwd.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vksdl {

class Allocator;
class Device;

// Standard interleaved vertex layout used by all model loaders.
// 32 bytes, tightly packed (no padding). All loaders convert to this format.
// Missing normals are generated as flat normals. Missing UVs default to (0,0).
// Note: v2 may extend this with tangents for normal mapping.
struct Vertex {
    float position[3];
    float normal[3];
    float texCoord[2];
};
static_assert(sizeof(Vertex) == 32, "Vertex layout changed -- update shaders");

// Basic PBR material values extracted from model files.
// Texture loading is the user's responsibility via loadImage().
struct Material {
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallic     = 0.0f;
    float roughness    = 1.0f;
    std::string name;
    std::filesystem::path diffuseTexture;  // empty if no texture; resolved against model dir
};

// CPU-side mesh data for one sub-mesh. Loaded by loadModel().
// Veterans access .vertices.data() and .indices.data() for custom upload via
// existing BufferBuilder + uploadToBuffer.
struct MeshData {
    std::vector<Vertex>        vertices;
    std::vector<std::uint32_t> indices;
    Material                   material;
    std::string                name;

    [[nodiscard]] VkDeviceSize vertexSizeBytes() const {
        return static_cast<VkDeviceSize>(vertices.size()) * sizeof(Vertex);
    }
    [[nodiscard]] VkDeviceSize indexSizeBytes() const {
        return static_cast<VkDeviceSize>(indices.size()) * sizeof(std::uint32_t);
    }
};

#if VKSDL_HAS_LOADERS

// CPU-side model data. Contains all meshes from a single file.
// Move-only (copying large vertex data is expensive and probably a bug).
struct ModelData {
    std::vector<MeshData> meshes;

    ModelData(ModelData&&) = default;
    ModelData& operator=(ModelData&&) = default;
    ModelData(const ModelData&) = delete;
    ModelData& operator=(const ModelData&) = delete;

private:
    friend Result<ModelData> loadModel(const std::filesystem::path&);
    ModelData() = default;
};

// Load a 3D model file. Supported formats: .gltf, .glb (glTF 2.0), .obj (Wavefront).
// Format detected by file extension. Returns meshes with interleaved Vertex data
// (position + normal + texCoord) and uint32 indices.
[[nodiscard]] Result<ModelData> loadModel(const std::filesystem::path& path);

#endif // VKSDL_HAS_LOADERS

// GPU-side mesh. Owns device-local vertex + index buffers via VMA.
// Created by uploadMesh(). Raw handle accessors for veteran-level Vulkan usage.
//
// Thread safety: immutable after construction.
class Mesh {
public:
    ~Mesh();
    Mesh(Mesh&&) noexcept;
    Mesh& operator=(Mesh&&) noexcept;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    [[nodiscard]] VkBuffer      vkVertexBuffer() const { return vertexBuffer_; }
    [[nodiscard]] VkBuffer      vkIndexBuffer()  const { return indexBuffer_; }
    [[nodiscard]] std::uint32_t vertexCount()    const { return vertexCount_; }
    [[nodiscard]] std::uint32_t indexCount()     const { return indexCount_; }

private:
    friend Result<Mesh> uploadMesh(const Allocator&, const Device&, const MeshData&);
    Mesh() = default;

    VmaAllocator  allocator_     = nullptr;
    VkBuffer      vertexBuffer_  = VK_NULL_HANDLE;
    VmaAllocation vertexAlloc_   = nullptr;
    VkBuffer      indexBuffer_   = VK_NULL_HANDLE;
    VmaAllocation indexAlloc_    = nullptr;
    std::uint32_t vertexCount_   = 0;
    std::uint32_t indexCount_    = 0;
};

// Staged upload: creates device-local vertex + index buffers from MeshData,
// copies data via staging buffers, waits for completion.
// Blocking -- suitable for init-time uploads only.
[[nodiscard]] Result<Mesh> uploadMesh(
    const Allocator& allocator,
    const Device& device,
    const MeshData& meshData);

} // namespace vksdl
