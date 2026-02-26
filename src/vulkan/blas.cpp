#include <vksdl/blas.hpp>
#include <vksdl/allocator.hpp>
#include <vksdl/buffer.hpp>
#include <vksdl/device.hpp>
#include <vksdl/mesh.hpp>
#include <vksdl/util.hpp>

#include "rt_functions.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

namespace vksdl {

struct Blas::BackingBuffer {
    VmaAllocator  allocator  = nullptr;
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};

Blas::~Blas() {
    // Destroy AS before its backing buffer.
    if (as_ != VK_NULL_HANDLE) {
        auto fn = detail::loadRtFunctions(device_);
        if (fn.destroyAs) {
            fn.destroyAs(device_, as_, nullptr);
        }
    }
    if (backing_) {
        if (backing_->buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(backing_->allocator, backing_->buffer,
                             backing_->allocation);
        }
        delete backing_;
    }
}

Blas::Blas(Blas&& o) noexcept
    : device_(o.device_), as_(o.as_), address_(o.address_),
      backing_(o.backing_) {
    o.device_  = VK_NULL_HANDLE;
    o.as_      = VK_NULL_HANDLE;
    o.address_ = 0;
    o.backing_ = nullptr;
}

Blas& Blas::operator=(Blas&& o) noexcept {
    if (this != &o) {
        if (as_ != VK_NULL_HANDLE) {
            auto fn = detail::loadRtFunctions(device_);
            if (fn.destroyAs) {
                fn.destroyAs(device_, as_, nullptr);
            }
        }
        if (backing_) {
            if (backing_->buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(backing_->allocator, backing_->buffer,
                                 backing_->allocation);
            }
            delete backing_;
        }

        device_  = o.device_;
        as_      = o.as_;
        address_ = o.address_;
        backing_ = o.backing_;

        o.device_  = VK_NULL_HANDLE;
        o.as_      = VK_NULL_HANDLE;
        o.address_ = 0;
        o.backing_ = nullptr;
    }
    return *this;
}

namespace {

VkAccelerationStructureGeometryKHR toVkGeometry(
    const BlasTriangleGeometry& g) {

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = g.vertexFormat;
    triangles.vertexData.deviceAddress = g.vertexBufferAddress;
    triangles.vertexStride  = g.vertexStride;
    triangles.maxVertex     = g.vertexCount > 0 ? g.vertexCount - 1 : 0;
    triangles.indexType     = g.indexType;
    triangles.indexData.deviceAddress = g.indexBufferAddress;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags        = g.opaque
        ? static_cast<VkGeometryFlagsKHR>(VK_GEOMETRY_OPAQUE_BIT_KHR)
        : 0;
    geom.geometry.triangles = triangles;

    return geom;
}

} // anonymous namespace

BlasTriangleGeometry BlasTriangleGeometry::fromBuffers(
    const Buffer& vertexBuffer, const Buffer& indexBuffer,
    std::uint32_t vertexCount, std::uint32_t indexCount,
    std::uint32_t vertexStride) {

    BlasTriangleGeometry geo{};
    geo.vertexBufferAddress = vertexBuffer.deviceAddress();
    geo.indexBufferAddress  = indexBuffer.deviceAddress();
    geo.vertexCount  = vertexCount;
    geo.indexCount   = indexCount;
    geo.vertexStride = vertexStride;
    return geo;
}

BlasBuilder::BlasBuilder(const Device& device, const Allocator& allocator)
    : device_(device.vkDevice()),
      physDevice_(device.vkPhysicalDevice()),
      queue_(device.graphicsQueue()),
      queueFamily_(device.queueFamilies().graphics),
      scratchAlignment_(device.minAccelerationStructureScratchOffsetAlignment()),
      allocator_(&allocator) {}

BlasBuilder& BlasBuilder::addTriangles(const BlasTriangleGeometry& geometry) {
    geometries_.push_back(geometry);
    return *this;
}

BlasBuilder& BlasBuilder::addMesh(const Mesh& mesh) {
    // Get device addresses via builder's VkDevice, not from Mesh.
    VkBufferDeviceAddressInfo vertexAddrInfo{};
    vertexAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    vertexAddrInfo.buffer = mesh.vkVertexBuffer();

    VkBufferDeviceAddressInfo indexAddrInfo{};
    indexAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    indexAddrInfo.buffer = mesh.vkIndexBuffer();

    BlasTriangleGeometry g{};
    g.vertexBufferAddress = vkGetBufferDeviceAddress(device_, &vertexAddrInfo);
    g.indexBufferAddress  = vkGetBufferDeviceAddress(device_, &indexAddrInfo);
    g.vertexCount         = mesh.vertexCount();
    g.indexCount          = mesh.indexCount();
    g.vertexStride        = sizeof(Vertex);
    g.vertexFormat        = VK_FORMAT_R32G32B32_SFLOAT;
    g.indexType           = VK_INDEX_TYPE_UINT32;
    g.opaque              = true;

    geometries_.push_back(g);
    return *this;
}

BlasBuilder& BlasBuilder::preferFastTrace() {
    buildFlags_ = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    return *this;
}

BlasBuilder& BlasBuilder::preferFastBuild() {
    buildFlags_ = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    return *this;
}

BlasBuilder& BlasBuilder::allowCompaction() {
    allowCompaction_ = true;
    return *this;
}

Result<BlasBuildSizes> BlasBuilder::sizes() const {
    if (geometries_.empty()) {
        return Error{"BLAS sizes", 0,
                     "no geometries added -- call addTriangles() or addMesh()"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.getBuildSizes) {
        return Error{"BLAS sizes", 0,
                     "vkGetAccelerationStructureBuildSizesKHR not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    std::vector<VkAccelerationStructureGeometryKHR> vkGeoms(geometries_.size());
    std::vector<std::uint32_t> maxPrimCounts(geometries_.size());
    for (std::size_t i = 0; i < geometries_.size(); ++i) {
        vkGeoms[i]       = toVkGeometry(geometries_[i]);
        maxPrimCounts[i] = geometries_[i].indexCount / 3;
    }

    VkBuildAccelerationStructureFlagsKHR flags = buildFlags_;
    if (allowCompaction_) {
        flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = flags;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = static_cast<std::uint32_t>(vkGeoms.size());
    buildInfo.pGeometries   = vkGeoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    fn.getBuildSizes(device_,
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, maxPrimCounts.data(), &sizeInfo);

    return BlasBuildSizes{
        sizeInfo.accelerationStructureSize,
        sizeInfo.buildScratchSize,
        sizeInfo.updateScratchSize,
    };
}

Result<Blas> BlasBuilder::build() {
    if (geometries_.empty()) {
        return Error{"BLAS build", 0,
                     "no geometries added -- call addTriangles() or addMesh()"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.createAs || !fn.cmdBuildAs || !fn.getBuildSizes ||
        !fn.getAsDeviceAddress || !fn.destroyAs) {
        return Error{"BLAS build", 0,
                     "RT extension functions not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    std::vector<VkAccelerationStructureGeometryKHR> vkGeoms(geometries_.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geometries_.size());
    std::vector<std::uint32_t> maxPrimCounts(geometries_.size());

    for (std::size_t i = 0; i < geometries_.size(); ++i) {
        vkGeoms[i]       = toVkGeometry(geometries_[i]);
        maxPrimCounts[i] = geometries_[i].indexCount / 3;

        ranges[i] = {};
        ranges[i].primitiveCount  = geometries_[i].indexCount / 3;
        ranges[i].primitiveOffset = 0;
        ranges[i].firstVertex     = 0;
        ranges[i].transformOffset = 0;
    }

    VkBuildAccelerationStructureFlagsKHR flags = buildFlags_;
    if (allowCompaction_) {
        flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = flags;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = static_cast<std::uint32_t>(vkGeoms.size());
    buildInfo.pGeometries   = vkGeoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    fn.getBuildSizes(device_,
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, maxPrimCounts.data(), &sizeInfo);

    // Raw VMA handles -- Blas owns the backing buffer directly to avoid
    // circular dependencies with the Buffer RAII type.
    VmaAllocator vma = allocator_->vmaAllocator();

    VkBufferCreateInfo backingCI{};
    backingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    backingCI.size  = sizeInfo.accelerationStructureSize;
    backingCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo backingAllocCI{};
    backingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    auto* backing = new Blas::BackingBuffer{};
    backing->allocator = vma;

    VkResult vr = vmaCreateBuffer(vma, &backingCI, &backingAllocCI,
                                   &backing->buffer, &backing->allocation,
                                   nullptr);
    if (vr != VK_SUCCESS) {
        delete backing;
        return Error{"BLAS build", static_cast<std::int32_t>(vr),
                     "failed to create acceleration structure buffer"};
    }

    // Set up before createAs so the destructor fires on any subsequent error.
    Blas blas;
    blas.device_  = device_;
    blas.backing_ = backing;

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = backing->buffer;
    asCI.offset = 0;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    vr = fn.createAs(device_, &asCI, nullptr, &blas.as_);
    if (vr != VK_SUCCESS) {
        return Error{"BLAS build", static_cast<std::int32_t>(vr),
                     "vkCreateAccelerationStructureKHR failed"};
    }

    // Pad for alignment: alignUp requires the buffer to be at least alignment-1 larger.
    VkDeviceSize scratchAllocSize = sizeInfo.buildScratchSize;
    if (scratchAlignment_ > 1) {
        scratchAllocSize += scratchAlignment_ - 1;
    }

    auto scratchResult = BufferBuilder(*allocator_)
        .scratchBuffer()
        .size(scratchAllocSize)
        .build();
    if (!scratchResult.ok()) {
        return scratchResult.error();
    }
    Buffer& scratch = scratchResult.value();

    VkDeviceAddress scratchAddr = scratch.deviceAddress();
    if (scratchAlignment_ > 1) {
        scratchAddr = alignUp(scratchAddr, scratchAlignment_);
    }

    buildInfo.dstAccelerationStructure = blas.as_;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (allowCompaction_) {
        VkQueryPoolCreateInfo queryPoolCI{};
        queryPoolCI.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCI.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        queryPoolCI.queryCount = 1;

        vr = vkCreateQueryPool(device_, &queryPoolCI, nullptr, &queryPool);
        if (vr != VK_SUCCESS) {
            return Error{"BLAS build", static_cast<std::int32_t>(vr),
                         "failed to create query pool for compaction"};
        }
    }

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = queueFamily_;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(device_, &poolCI, nullptr, &cmdPool);
    if (vr != VK_SUCCESS) {
        if (queryPool) vkDestroyQueryPool(device_, queryPool, nullptr);
        return Error{"BLAS build", static_cast<std::int32_t>(vr),
                     "failed to create command pool"};
    }

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool        = cmdPool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(device_, &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device_, cmdPool, nullptr);
        if (queryPool) vkDestroyQueryPool(device_, queryPool, nullptr);
        return Error{"BLAS build", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
    fn.cmdBuildAs(cmd, 1, &buildInfo, &pRanges);

    // Compacted-size query in the same submit as the build.
    // Barrier ensures the build completes before the property read.
    if (allowCompaction_) {
        VkMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo depInfo{};
        depInfo.sType                = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount   = 1;
        depInfo.pMemoryBarriers      = &barrier;

        vkCmdPipelineBarrier2(cmd, &depInfo);

        vkCmdResetQueryPool(cmd, queryPool, 0, 1);
        fn.cmdWriteAsProperties(
            cmd, 1, &blas.as_,
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            queryPool, 0);
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vr = vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device_, cmdPool, nullptr);
        if (queryPool) vkDestroyQueryPool(device_, queryPool, nullptr);
        return Error{"BLAS build", static_cast<std::int32_t>(vr),
                     "failed to submit build command"};
    }

    // VKSDL_BLOCKING_WAIT: synchronous BLAS build waits for completion.
    vkQueueWaitIdle(queue_);
    vkDestroyCommandPool(device_, cmdPool, nullptr);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blas.as_;
    blas.address_ = fn.getAsDeviceAddress(device_, &addrInfo);

    // Compaction: read compacted size, create smaller copy, swap.
    if (allowCompaction_) {
        VkDeviceSize compactedSize = 0;
        vr = vkGetQueryPoolResults(
            device_, queryPool, 0, 1,
            sizeof(compactedSize), &compactedSize,
            sizeof(compactedSize),
            // VKSDL_BLOCKING_WAIT: synchronous compaction path waits for query results.
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        if (vr != VK_SUCCESS || compactedSize == 0) {
            vkDestroyQueryPool(device_, queryPool, nullptr);
            return Error{"BLAS compact", static_cast<std::int32_t>(vr),
                         "failed to read compacted size"};
        }

        VkBufferCreateInfo compactBackingCI{};
        compactBackingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        compactBackingCI.size  = compactedSize;
        compactBackingCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        auto* compactBacking = new Blas::BackingBuffer{};
        compactBacking->allocator = vma;

        vr = vmaCreateBuffer(vma, &compactBackingCI, &backingAllocCI,
                              &compactBacking->buffer,
                              &compactBacking->allocation, nullptr);
        if (vr != VK_SUCCESS) {
            delete compactBacking;
            vkDestroyQueryPool(device_, queryPool, nullptr);
            return Error{"BLAS compact", static_cast<std::int32_t>(vr),
                         "failed to create compacted buffer"};
        }

        VkAccelerationStructureCreateInfoKHR compactAsCI{};
        compactAsCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        compactAsCI.buffer = compactBacking->buffer;
        compactAsCI.offset = 0;
        compactAsCI.size   = compactedSize;
        compactAsCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        VkAccelerationStructureKHR compactAs = VK_NULL_HANDLE;
        vr = fn.createAs(device_, &compactAsCI, nullptr, &compactAs);
        if (vr != VK_SUCCESS) {
            delete compactBacking;
            vkDestroyQueryPool(device_, queryPool, nullptr);
            return Error{"BLAS compact", static_cast<std::int32_t>(vr),
                         "vkCreateAccelerationStructureKHR failed (compact)"};
        }

        VkCommandPool cmdPool2 = VK_NULL_HANDLE;
        vr = vkCreateCommandPool(device_, &poolCI, nullptr, &cmdPool2);
        if (vr != VK_SUCCESS) {
            fn.destroyAs(device_, compactAs, nullptr);
            delete compactBacking;
            vkDestroyQueryPool(device_, queryPool, nullptr);
            return Error{"BLAS compact", static_cast<std::int32_t>(vr),
                         "failed to create command pool for compaction"};
        }

        VkCommandBufferAllocateInfo cmdAI2{};
        cmdAI2.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAI2.commandPool        = cmdPool2;
        cmdAI2.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAI2.commandBufferCount = 1;

        VkCommandBuffer cmd2 = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(device_, &cmdAI2, &cmd2);
        if (vr != VK_SUCCESS) {
            vkDestroyCommandPool(device_, cmdPool2, nullptr);
            fn.destroyAs(device_, compactAs, nullptr);
            delete compactBacking;
            vkDestroyQueryPool(device_, queryPool, nullptr);
            return Error{"BLAS compact", static_cast<std::int32_t>(vr),
                         "failed to allocate command buffer for compaction"};
        }

        VkCommandBufferBeginInfo beginInfo2{};
        beginInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd2, &beginInfo2);

        VkCopyAccelerationStructureInfoKHR copyInfo{};
        copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
        copyInfo.src   = blas.as_;
        copyInfo.dst   = compactAs;
        copyInfo.mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

        fn.cmdCopyAs(cmd2, &copyInfo);
        vkEndCommandBuffer(cmd2);

        VkSubmitInfo submitInfo2{};
        submitInfo2.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo2.commandBufferCount = 1;
        submitInfo2.pCommandBuffers    = &cmd2;

        vr = vkQueueSubmit(queue_, 1, &submitInfo2, VK_NULL_HANDLE);
        if (vr != VK_SUCCESS) {
            vkDestroyCommandPool(device_, cmdPool2, nullptr);
            fn.destroyAs(device_, compactAs, nullptr);
            delete compactBacking;
            vkDestroyQueryPool(device_, queryPool, nullptr);
            return Error{"BLAS compact", static_cast<std::int32_t>(vr),
                         "failed to submit compaction copy"};
        }

        // VKSDL_BLOCKING_WAIT: synchronous BLAS compaction copy waits for completion.
        vkQueueWaitIdle(queue_);
        vkDestroyCommandPool(device_, cmdPool2, nullptr);

        fn.destroyAs(device_, blas.as_, nullptr);
        vmaDestroyBuffer(backing->allocator, backing->buffer,
                         backing->allocation);
        delete backing;

        blas.as_      = compactAs;
        blas.backing_ = compactBacking;

        // Re-query: compaction copy produces a new AS object with a new address.
        addrInfo.accelerationStructure = compactAs;
        blas.address_ = fn.getAsDeviceAddress(device_, &addrInfo);

        vkDestroyQueryPool(device_, queryPool, nullptr);
    }

    return blas;
}

Result<Blas> BlasBuilder::cmdBuild(VkCommandBuffer cmd, const Buffer& scratch) {
    if (geometries_.empty()) {
        return Error{"BLAS cmdBuild", 0,
                     "no geometries added -- call addTriangles() or addMesh()"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.createAs || !fn.cmdBuildAs || !fn.getBuildSizes ||
        !fn.getAsDeviceAddress) {
        return Error{"BLAS cmdBuild", 0,
                     "RT extension functions not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    std::vector<VkAccelerationStructureGeometryKHR> vkGeoms(geometries_.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geometries_.size());
    std::vector<std::uint32_t> maxPrimCounts(geometries_.size());

    for (std::size_t i = 0; i < geometries_.size(); ++i) {
        vkGeoms[i]       = toVkGeometry(geometries_[i]);
        maxPrimCounts[i] = geometries_[i].indexCount / 3;

        ranges[i] = {};
        ranges[i].primitiveCount  = geometries_[i].indexCount / 3;
        ranges[i].primitiveOffset = 0;
        ranges[i].firstVertex     = 0;
        ranges[i].transformOffset = 0;
    }

    VkBuildAccelerationStructureFlagsKHR flags = buildFlags_;
    if (allowCompaction_) {
        flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = flags;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = static_cast<std::uint32_t>(vkGeoms.size());
    buildInfo.pGeometries   = vkGeoms.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    fn.getBuildSizes(device_,
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, maxPrimCounts.data(), &sizeInfo);

    VmaAllocator vma = allocator_->vmaAllocator();

    VkBufferCreateInfo backingCI{};
    backingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    backingCI.size  = sizeInfo.accelerationStructureSize;
    backingCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo backingAllocCI{};
    backingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    auto* backing = new Blas::BackingBuffer{};
    backing->allocator = vma;

    VkResult vr = vmaCreateBuffer(vma, &backingCI, &backingAllocCI,
                                   &backing->buffer, &backing->allocation,
                                   nullptr);
    if (vr != VK_SUCCESS) {
        delete backing;
        return Error{"BLAS cmdBuild", static_cast<std::int32_t>(vr),
                     "failed to create acceleration structure buffer"};
    }

    Blas blas;
    blas.device_  = device_;
    blas.backing_ = backing;

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = backing->buffer;
    asCI.offset = 0;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    vr = fn.createAs(device_, &asCI, nullptr, &blas.as_);
    if (vr != VK_SUCCESS) {
        return Error{"BLAS cmdBuild", static_cast<std::int32_t>(vr),
                     "vkCreateAccelerationStructureKHR failed"};
    }

    VkDeviceAddress scratchAddr = scratch.deviceAddress();
    if (scratchAlignment_ > 1) {
        scratchAddr = alignUp(scratchAddr, scratchAlignment_);
    }

    buildInfo.dstAccelerationStructure  = blas.as_;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
    fn.cmdBuildAs(cmd, 1, &buildInfo, &pRanges);

    // Device address is valid immediately after AS creation (before build
    // executes). The build populates the data; the address is the memory
    // location, which is fixed at creation.
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blas.as_;
    blas.address_ = fn.getAsDeviceAddress(device_, &addrInfo);

    return blas;
}

Result<void> compactBlas(
    const Device& device, const Allocator& allocator, Blas& blas) {

    // Precondition: the BLAS must have been built with ALLOW_COMPACTION_BIT.
    // The Vulkan spec requires this flag for compacted-size queries to return
    // valid results. We cannot check this at runtime because Blas does not
    // store its original build flags.
    if (blas.as_ == VK_NULL_HANDLE) {
        return Error{"compact BLAS", 0, "BLAS has no acceleration structure"};
    }

    VkDevice dev   = device.vkDevice();
    VkQueue  queue = device.graphicsQueue();
    std::uint32_t family = device.queueFamilies().graphics;

    auto fn = detail::loadRtFunctions(dev);
    if (!fn.createAs || !fn.destroyAs || !fn.cmdWriteAsProperties ||
        !fn.cmdCopyAs || !fn.getAsDeviceAddress) {
        return Error{"compact BLAS", 0,
                     "RT extension functions not available"};
    }

    VmaAllocator vma = allocator.vmaAllocator();

    VkQueryPoolCreateInfo queryPoolCI{};
    queryPoolCI.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCI.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    queryPoolCI.queryCount = 1;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    VkResult vr = vkCreateQueryPool(dev, &queryPoolCI, nullptr, &queryPool);
    if (vr != VK_SUCCESS) {
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to create query pool"};
    }

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = family;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(dev, &poolCI, nullptr, &cmdPool);
    if (vr != VK_SUCCESS) {
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to create command pool"};
    }

    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool        = cmdPool;
    cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(dev, &cmdAI, &cmd);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(dev, cmdPool, nullptr);
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdResetQueryPool(cmd, queryPool, 0, 1);
    fn.cmdWriteAsProperties(
        cmd, 1, &blas.as_,
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
        queryPool, 0);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vr = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(dev, cmdPool, nullptr);
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to submit property query"};
    }

    // VKSDL_BLOCKING_WAIT: helper compaction query waits for transfer queue idle.
    vkQueueWaitIdle(queue);
    vkDestroyCommandPool(dev, cmdPool, nullptr);

    VkDeviceSize compactedSize = 0;
    vr = vkGetQueryPoolResults(
        dev, queryPool, 0, 1,
        sizeof(compactedSize), &compactedSize,
        sizeof(compactedSize),
        // VKSDL_BLOCKING_WAIT: helper compaction path waits for query results.
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (vr != VK_SUCCESS || compactedSize == 0) {
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to read compacted size"};
    }

    VkBufferCreateInfo compactCI{};
    compactCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    compactCI.size  = compactedSize;
    compactCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo compactAllocCI{};
    compactAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    auto* compactBacking = new Blas::BackingBuffer{};
    compactBacking->allocator = vma;

    vr = vmaCreateBuffer(vma, &compactCI, &compactAllocCI,
                          &compactBacking->buffer,
                          &compactBacking->allocation, nullptr);
    if (vr != VK_SUCCESS) {
        delete compactBacking;
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to create compacted buffer"};
    }

    VkAccelerationStructureCreateInfoKHR compactAsCI{};
    compactAsCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    compactAsCI.buffer = compactBacking->buffer;
    compactAsCI.offset = 0;
    compactAsCI.size   = compactedSize;
    compactAsCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR compactAs = VK_NULL_HANDLE;
    vr = fn.createAs(dev, &compactAsCI, nullptr, &compactAs);
    if (vr != VK_SUCCESS) {
        delete compactBacking;
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "vkCreateAccelerationStructureKHR failed (compact)"};
    }

    VkCommandPool cmdPool2 = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(dev, &poolCI, nullptr, &cmdPool2);
    if (vr != VK_SUCCESS) {
        fn.destroyAs(dev, compactAs, nullptr);
        delete compactBacking;
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to create command pool for copy"};
    }

    VkCommandBufferAllocateInfo cmdAI2{};
    cmdAI2.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI2.commandPool        = cmdPool2;
    cmdAI2.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI2.commandBufferCount = 1;

    VkCommandBuffer cmd2 = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(dev, &cmdAI2, &cmd2);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(dev, cmdPool2, nullptr);
        fn.destroyAs(dev, compactAs, nullptr);
        delete compactBacking;
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer for copy"};
    }

    VkCommandBufferBeginInfo beginInfo2{};
    beginInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd2, &beginInfo2);

    VkCopyAccelerationStructureInfoKHR copyInfo{};
    copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
    copyInfo.src   = blas.as_;
    copyInfo.dst   = compactAs;
    copyInfo.mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

    fn.cmdCopyAs(cmd2, &copyInfo);
    vkEndCommandBuffer(cmd2);

    VkSubmitInfo submitInfo2{};
    submitInfo2.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo2.commandBufferCount = 1;
    submitInfo2.pCommandBuffers    = &cmd2;

    vr = vkQueueSubmit(queue, 1, &submitInfo2, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(dev, cmdPool2, nullptr);
        fn.destroyAs(dev, compactAs, nullptr);
        delete compactBacking;
        vkDestroyQueryPool(dev, queryPool, nullptr);
        return Error{"compact BLAS", static_cast<std::int32_t>(vr),
                     "failed to submit compaction copy"};
    }

    // VKSDL_BLOCKING_WAIT: helper compaction copy waits for completion.
    vkQueueWaitIdle(queue);
    vkDestroyCommandPool(dev, cmdPool2, nullptr);
    vkDestroyQueryPool(dev, queryPool, nullptr);

    fn.destroyAs(dev, blas.as_, nullptr);
    if (blas.backing_->buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(blas.backing_->allocator, blas.backing_->buffer,
                         blas.backing_->allocation);
    }
    delete blas.backing_;

    blas.as_      = compactAs;
    blas.backing_ = compactBacking;

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = compactAs;
    blas.address_ = fn.getAsDeviceAddress(dev, &addrInfo);

    return {};
}

} // namespace vksdl
