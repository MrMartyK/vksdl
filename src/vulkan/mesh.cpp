#include <vksdl/mesh.hpp>
#include <vksdl/allocator.hpp>
#include <vksdl/device.hpp>

#if VKSDL_HAS_LOADERS
#include "mesh_loaders.hpp"
#include <algorithm>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

#include <cstring>

namespace vksdl {

#if VKSDL_HAS_LOADERS

Result<ModelData> loadModel(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    Result<std::vector<MeshData>> meshes = Error{"load model", 0, ""};

    if (ext == ".gltf" || ext == ".glb") {
        meshes = detail::loadGltf(path);
    } else if (ext == ".obj") {
        meshes = detail::loadObj(path);
    } else {
        return Error{"load model", 0,
                     "unsupported model format '" + ext +
                     "' -- supported: .gltf, .glb, .obj"};
    }

    if (!meshes.ok()) {
        return meshes.error();
    }

    ModelData model;
    model.meshes = std::move(meshes.value());
    return model;
}

#endif // VKSDL_HAS_LOADERS

Mesh::~Mesh() {
    if (allocator_) {
        if (indexBuffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, indexBuffer_, indexAlloc_);
        }
        if (vertexBuffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, vertexBuffer_, vertexAlloc_);
        }
    }
}

Mesh::Mesh(Mesh&& o) noexcept
    : allocator_(o.allocator_), vertexBuffer_(o.vertexBuffer_),
      vertexAlloc_(o.vertexAlloc_), indexBuffer_(o.indexBuffer_),
      indexAlloc_(o.indexAlloc_), vertexCount_(o.vertexCount_),
      indexCount_(o.indexCount_) {
    o.allocator_    = nullptr;
    o.vertexBuffer_ = VK_NULL_HANDLE;
    o.vertexAlloc_  = nullptr;
    o.indexBuffer_  = VK_NULL_HANDLE;
    o.indexAlloc_   = nullptr;
    o.vertexCount_  = 0;
    o.indexCount_   = 0;
}

Mesh& Mesh::operator=(Mesh&& o) noexcept {
    if (this != &o) {
        if (allocator_) {
            if (indexBuffer_ != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator_, indexBuffer_, indexAlloc_);
            }
            if (vertexBuffer_ != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator_, vertexBuffer_, vertexAlloc_);
            }
        }
        allocator_    = o.allocator_;
        vertexBuffer_ = o.vertexBuffer_;
        vertexAlloc_  = o.vertexAlloc_;
        indexBuffer_  = o.indexBuffer_;
        indexAlloc_   = o.indexAlloc_;
        vertexCount_  = o.vertexCount_;
        indexCount_   = o.indexCount_;
        o.allocator_    = nullptr;
        o.vertexBuffer_ = VK_NULL_HANDLE;
        o.vertexAlloc_  = nullptr;
        o.indexBuffer_  = VK_NULL_HANDLE;
        o.indexAlloc_   = nullptr;
        o.vertexCount_  = 0;
        o.indexCount_   = 0;
    }
    return *this;
}

Result<Mesh> uploadMesh(
    const Allocator& allocator,
    const Device& device,
    const MeshData& meshData) {

    if (meshData.vertices.empty()) {
        return Error{"upload mesh", 0, "MeshData has no vertices"};
    }
    if (meshData.indices.empty()) {
        return Error{"upload mesh", 0, "MeshData has no indices"};
    }

    VkDeviceSize vertexSize = meshData.vertexSizeBytes();
    VkDeviceSize indexSize  = meshData.indexSizeBytes();
    VkDeviceSize totalSize  = vertexSize + indexSize;

    VmaAllocator vma = allocator.vmaAllocator();

    // Create device-local vertex buffer. SHADER_DEVICE_ADDRESS_BIT included
    // so Mesh objects work directly with BlasBuilder::addMesh() for RT.
    VkBufferCreateInfo vertexCI{};
    vertexCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertexCI.size  = vertexSize;
    vertexCI.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo deviceAllocCI{};
    deviceAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    Mesh mesh;
    mesh.allocator_   = vma;
    mesh.vertexCount_ = static_cast<std::uint32_t>(meshData.vertices.size());
    mesh.indexCount_  = static_cast<std::uint32_t>(meshData.indices.size());

    VkResult vr = vmaCreateBuffer(vma, &vertexCI, &deviceAllocCI,
                                   &mesh.vertexBuffer_, &mesh.vertexAlloc_,
                                   nullptr);
    if (vr != VK_SUCCESS) {
        return Error{"upload mesh", static_cast<std::int32_t>(vr),
                     "failed to create vertex buffer"};
    }

    VkBufferCreateInfo indexCI{};
    indexCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    indexCI.size  = indexSize;
    indexCI.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    vr = vmaCreateBuffer(vma, &indexCI, &deviceAllocCI,
                          &mesh.indexBuffer_, &mesh.indexAlloc_, nullptr);
    if (vr != VK_SUCCESS) {
        return Error{"upload mesh", static_cast<std::int32_t>(vr),
                     "failed to create index buffer"};
    }

    VkBufferCreateInfo stagingCI{};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size  = totalSize;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI{};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer      stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo stagingInfo{};

    vr = vmaCreateBuffer(vma, &stagingCI, &stagingAllocCI,
                          &stagingBuf, &stagingAlloc, &stagingInfo);
    if (vr != VK_SUCCESS) {
        return Error{"upload mesh", static_cast<std::int32_t>(vr),
                     "failed to create staging buffer"};
    }

    auto* dst = static_cast<unsigned char*>(stagingInfo.pMappedData);
    std::memcpy(dst, meshData.vertices.data(),
                static_cast<std::size_t>(vertexSize));
    std::memcpy(dst + vertexSize, meshData.indices.data(),
                static_cast<std::size_t>(indexSize));

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = device.queueFamilies().graphics;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(device.vkDevice(), &poolCI, nullptr, &cmdPool);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"upload mesh", static_cast<std::int32_t>(vr),
                     "failed to create command pool for transfer"};
    }

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool        = cmdPool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(device.vkDevice(), &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"upload mesh", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer for transfer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy vertexRegion{};
    vertexRegion.srcOffset = 0;
    vertexRegion.dstOffset = 0;
    vertexRegion.size      = vertexSize;
    vkCmdCopyBuffer(cmd, stagingBuf, mesh.vertexBuffer_, 1, &vertexRegion);

    VkBufferCopy indexRegion{};
    indexRegion.srcOffset = vertexSize;
    indexRegion.dstOffset = 0;
    indexRegion.size      = indexSize;
    vkCmdCopyBuffer(cmd, stagingBuf, mesh.indexBuffer_, 1, &indexRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vr = vkQueueSubmit(device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"upload mesh", static_cast<std::int32_t>(vr),
                     "vkQueueSubmit failed for mesh transfer"};
    }
    // VKSDL_BLOCKING_WAIT: init-time mesh upload waits for transfer completion.
    vkQueueWaitIdle(device.graphicsQueue());

    vkDestroyCommandPool(device.vkDevice(), cmdPool, nullptr);
    vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);

    return mesh;
}

} // namespace vksdl
