#include <vksdl/tlas.hpp>
#include <vksdl/allocator.hpp>
#include <vksdl/blas.hpp>
#include <vksdl/buffer.hpp>
#include <vksdl/device.hpp>
#include <vksdl/util.hpp>

#include "rt_functions.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <vk_mem_alloc.h>
#pragma GCC diagnostic pop

#include <cstring>

namespace vksdl {

struct Tlas::BackingBuffer {
    VmaAllocator  allocator  = nullptr;
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};

Tlas::~Tlas() {
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
    if (instance_) {
        if (instance_->buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(instance_->allocator, instance_->buffer,
                             instance_->allocation);
        }
        delete instance_;
    }
}

Tlas::Tlas(Tlas&& o) noexcept
    : device_(o.device_)
    , as_(o.as_)
    , allowUpdate_(o.allowUpdate_)
    , maxInstanceCount_(o.maxInstanceCount_)
    , scratchAlignment_(o.scratchAlignment_)
    , buildFlags_(o.buildFlags_)
    , updateScratchSize_(o.updateScratchSize_)
    , backing_(o.backing_)
    , instance_(o.instance_) {
    o.device_   = VK_NULL_HANDLE;
    o.as_       = VK_NULL_HANDLE;
    o.allowUpdate_ = false;
    o.maxInstanceCount_ = 0;
    o.scratchAlignment_ = 0;
    o.buildFlags_ = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    o.updateScratchSize_ = 0;
    o.backing_  = nullptr;
    o.instance_ = nullptr;
}

Tlas& Tlas::operator=(Tlas&& o) noexcept {
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
        if (instance_) {
            if (instance_->buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(instance_->allocator, instance_->buffer,
                                 instance_->allocation);
            }
            delete instance_;
        }

        device_   = o.device_;
        as_       = o.as_;
        allowUpdate_ = o.allowUpdate_;
        maxInstanceCount_ = o.maxInstanceCount_;
        scratchAlignment_ = o.scratchAlignment_;
        buildFlags_ = o.buildFlags_;
        updateScratchSize_ = o.updateScratchSize_;
        backing_  = o.backing_;
        instance_ = o.instance_;

        o.device_   = VK_NULL_HANDLE;
        o.as_       = VK_NULL_HANDLE;
        o.allowUpdate_ = false;
        o.maxInstanceCount_ = 0;
        o.scratchAlignment_ = 0;
        o.buildFlags_ = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        o.updateScratchSize_ = 0;
        o.backing_  = nullptr;
        o.instance_ = nullptr;
    }
    return *this;
}

namespace {

VkAccelerationStructureInstanceKHR toVkInstance(const TlasInstance& inst) {
    VkAccelerationStructureInstanceKHR vkInst{};

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            vkInst.transform.matrix[r][c] = inst.transform[r][c];
        }
    }

    vkInst.instanceCustomIndex                    = inst.customIndex & 0x00FFFFFF;
    vkInst.mask                                   = inst.mask;
    vkInst.instanceShaderBindingTableRecordOffset = inst.sbtOffset & 0x00FFFFFF;
    vkInst.flags                                  = inst.flags & 0xFF;
    vkInst.accelerationStructureReference         = inst.blasAddress;

    return vkInst;
}

} // anonymous namespace

TlasBuilder::TlasBuilder(const Device& device, const Allocator& allocator)
    : device_(device.vkDevice()),
      physDevice_(device.vkPhysicalDevice()),
      queue_(device.graphicsQueue()),
      queueFamily_(device.queueFamilies().graphics),
      scratchAlignment_(device.minAccelerationStructureScratchOffsetAlignment()),
      allocator_(&allocator) {}

TlasBuilder& TlasBuilder::addInstance(
    const Blas& blas, const float transform[3][4],
    std::uint32_t customIndex, std::uint8_t mask,
    std::uint32_t sbtOffset) {

    TlasInstance inst{};
    inst.blasAddress = blas.deviceAddress();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            inst.transform[r][c] = transform[r][c];
        }
    }
    inst.customIndex = customIndex;
    inst.mask        = mask;
    inst.sbtOffset   = sbtOffset;

    instances_.push_back(inst);
    return *this;
}

TlasBuilder& TlasBuilder::addInstance(const TlasInstance& instance) {
    instances_.push_back(instance);
    return *this;
}

TlasBuilder& TlasBuilder::preferFastTrace() {
    buildFlags_ = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    return *this;
}

TlasBuilder& TlasBuilder::preferFastBuild() {
    buildFlags_ = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    return *this;
}

TlasBuilder& TlasBuilder::allowUpdate() {
    allowUpdate_ = true;
    return *this;
}

Result<Tlas> TlasBuilder::build() {
    if (instances_.empty()) {
        return Error{"TLAS build", 0,
                     "no instances added -- call addInstance()"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.createAs || !fn.cmdBuildAs || !fn.getBuildSizes || !fn.destroyAs) {
        return Error{"TLAS build", 0,
                     "RT extension functions not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    VmaAllocator vma = allocator_->vmaAllocator();

    std::vector<VkAccelerationStructureInstanceKHR> vkInstances(instances_.size());
    for (std::size_t i = 0; i < instances_.size(); ++i) {
        vkInstances[i] = toVkInstance(instances_[i]);
    }

    VkDeviceSize instanceDataSize =
        static_cast<VkDeviceSize>(vkInstances.size()) *
        sizeof(VkAccelerationStructureInstanceKHR);

    VkBufferCreateInfo instanceCI{};
    instanceCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    instanceCI.size  = instanceDataSize;
    instanceCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo instanceAllocCI{};
    instanceAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    auto* instanceBuf = new Tlas::BackingBuffer{};
    instanceBuf->allocator = vma;

    VkResult vr = vmaCreateBuffer(vma, &instanceCI, &instanceAllocCI,
                                   &instanceBuf->buffer, &instanceBuf->allocation,
                                   nullptr);
    if (vr != VK_SUCCESS) {
        delete instanceBuf;
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
                     "failed to create instance buffer"};
    }

    VkBufferCreateInfo stagingCI{};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size  = instanceDataSize;
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
        delete instanceBuf;
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
                     "failed to create staging buffer for instances"};
    }

    std::memcpy(stagingInfo.pMappedData, vkInstances.data(),
                static_cast<std::size_t>(instanceDataSize));

    VkBufferDeviceAddressInfo instanceAddrInfo{};
    instanceAddrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    instanceAddrInfo.buffer = instanceBuf->buffer;
    VkDeviceAddress instanceAddr = vkGetBufferDeviceAddress(device_, &instanceAddrInfo);

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceAddr;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = instancesData;

    VkBuildAccelerationStructureFlagsKHR flags = buildFlags_;
    if (allowUpdate_) {
        flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }

    auto instanceCount = static_cast<std::uint32_t>(instances_.size());

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = flags;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    fn.getBuildSizes(device_,
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, &instanceCount, &sizeInfo);

    VkBufferCreateInfo backingCI{};
    backingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    backingCI.size  = sizeInfo.accelerationStructureSize;
    backingCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo backingAllocCI{};
    backingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    auto* backing = new Tlas::BackingBuffer{};
    backing->allocator = vma;

    vr = vmaCreateBuffer(vma, &backingCI, &backingAllocCI,
                          &backing->buffer, &backing->allocation, nullptr);
    if (vr != VK_SUCCESS) {
        delete backing;
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        delete instanceBuf;
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
                     "failed to create acceleration structure buffer"};
    }

    // Set up before createAs so the destructor fires on any subsequent error.
    Tlas tlas;
    tlas.device_   = device_;
    tlas.allowUpdate_ = allowUpdate_;
    tlas.maxInstanceCount_ = instanceCount;
    tlas.scratchAlignment_ = scratchAlignment_;
    tlas.buildFlags_ = flags;
    tlas.updateScratchSize_ = sizeInfo.updateScratchSize;
    if (scratchAlignment_ > 1) {
        tlas.updateScratchSize_ += scratchAlignment_ - 1;
    }
    tlas.backing_  = backing;
    tlas.instance_ = instanceBuf;

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = backing->buffer;
    asCI.offset = 0;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    vr = fn.createAs(device_, &asCI, nullptr, &tlas.as_);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
                     "vkCreateAccelerationStructureKHR failed"};
    }

    VkDeviceSize scratchAllocSize = sizeInfo.buildScratchSize;
    if (scratchAlignment_ > 1) {
        scratchAllocSize += scratchAlignment_ - 1;
    }

    auto scratchResult = BufferBuilder(*allocator_)
        .scratchBuffer()
        .size(scratchAllocSize)
        .build();
    if (!scratchResult.ok()) {
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return scratchResult.error();
    }
    Buffer& scratch = scratchResult.value();

    VkDeviceAddress scratchAddr = scratch.deviceAddress();
    if (scratchAlignment_ > 1) {
        scratchAddr = alignUp(scratchAddr, scratchAlignment_);
    }

    buildInfo.dstAccelerationStructure  = tlas.as_;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = queueFamily_;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vr = vkCreateCommandPool(device_, &poolCI, nullptr, &cmdPool);
    if (vr != VK_SUCCESS) {
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
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
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
                     "failed to allocate command buffer"};
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = instanceDataSize;
    vkCmdCopyBuffer(cmd, stagingBuf, instanceBuf->buffer, 1, &copyRegion);

    // Barrier: transfer write -> AS build read.
    VkMemoryBarrier2 transferBarrier{};
    transferBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    transferBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    transferBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    transferBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    transferBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo depInfo{};
    depInfo.sType                = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount   = 1;
    depInfo.pMemoryBarriers      = &transferBarrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;

    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &rangeInfo;
    fn.cmdBuildAs(cmd, 1, &buildInfo, &pRanges);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vr = vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(device_, cmdPool, nullptr);
        vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);
        return Error{"TLAS build", static_cast<std::int32_t>(vr),
                     "failed to submit build command"};
    }

    vkQueueWaitIdle(queue_);
    vkDestroyCommandPool(device_, cmdPool, nullptr);
    vmaDestroyBuffer(vma, stagingBuf, stagingAlloc);

    return tlas;
}

Result<Tlas> TlasBuilder::cmdBuild(VkCommandBuffer cmd,
                                    const Buffer& scratch,
                                    const Buffer& instanceBuffer,
                                    std::uint32_t instanceCount) {
    if (instanceCount == 0) {
        return Error{"TLAS cmdBuild", 0, "instanceCount is 0"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.createAs || !fn.cmdBuildAs || !fn.getBuildSizes) {
        return Error{"TLAS cmdBuild", 0,
                     "RT extension functions not available "
                     "-- did you call needRayTracingPipeline()?"};
    }

    VmaAllocator vma = allocator_->vmaAllocator();

    VkDeviceAddress instanceAddr = instanceBuffer.deviceAddress();

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceAddr;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = instancesData;

    VkBuildAccelerationStructureFlagsKHR flags = buildFlags_;
    if (allowUpdate_) {
        flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = flags;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    fn.getBuildSizes(device_,
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &buildInfo, &instanceCount, &sizeInfo);

    VkBufferCreateInfo backingCI{};
    backingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    backingCI.size  = sizeInfo.accelerationStructureSize;
    backingCI.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo backingAllocCI{};
    backingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;

    auto* backing = new Tlas::BackingBuffer{};
    backing->allocator = vma;

    VkResult vr = vmaCreateBuffer(vma, &backingCI, &backingAllocCI,
                                   &backing->buffer, &backing->allocation,
                                   nullptr);
    if (vr != VK_SUCCESS) {
        delete backing;
        return Error{"TLAS cmdBuild", static_cast<std::int32_t>(vr),
                     "failed to create acceleration structure buffer"};
    }

    Tlas tlas;
    tlas.device_  = device_;
    tlas.allowUpdate_ = allowUpdate_;
    tlas.maxInstanceCount_ = instanceCount;
    tlas.scratchAlignment_ = scratchAlignment_;
    tlas.buildFlags_ = flags;
    tlas.updateScratchSize_ = sizeInfo.updateScratchSize;
    if (scratchAlignment_ > 1) {
        tlas.updateScratchSize_ += scratchAlignment_ - 1;
    }
    tlas.backing_ = backing;
    // instance_ is nullptr: user owns the instance buffer in async path.

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = backing->buffer;
    asCI.offset = 0;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    vr = fn.createAs(device_, &asCI, nullptr, &tlas.as_);
    if (vr != VK_SUCCESS) {
        return Error{"TLAS cmdBuild", static_cast<std::int32_t>(vr),
                     "vkCreateAccelerationStructureKHR failed"};
    }

    VkDeviceAddress scratchAddr = scratch.deviceAddress();
    if (scratchAlignment_ > 1) {
        scratchAddr = alignUp(scratchAddr, scratchAlignment_);
    }

    buildInfo.dstAccelerationStructure  = tlas.as_;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;

    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &rangeInfo;
    fn.cmdBuildAs(cmd, 1, &buildInfo, &pRanges);

    return tlas;
}

Result<void> Tlas::cmdUpdate(VkCommandBuffer cmd,
                             const Buffer& scratch,
                             const Buffer& instanceBuffer,
                             std::uint32_t instanceCount) {
    if (!allowUpdate_) {
        return Error{"TLAS cmdUpdate", 0,
                     "TLAS was not built with allowUpdate()"};
    }
    if (as_ == VK_NULL_HANDLE) {
        return Error{"TLAS cmdUpdate", 0, "TLAS handle is null"};
    }
    if (instanceCount == 0) {
        return Error{"TLAS cmdUpdate", 0, "instanceCount is 0"};
    }
    if (instanceCount > maxInstanceCount_) {
        return Error{"TLAS cmdUpdate", 0,
                     "instanceCount exceeds TLAS maxInstanceCount"};
    }
    if (scratch.size() < updateScratchSize_) {
        return Error{"TLAS cmdUpdate", 0,
                     "scratch buffer is smaller than updateScratchSize"};
    }

    auto fn = detail::loadRtFunctions(device_);
    if (!fn.cmdBuildAs) {
        return Error{"TLAS cmdUpdate", 0,
                     "RT extension functions not available"};
    }

    VkDeviceAddress instanceAddr = instanceBuffer.deviceAddress();

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceAddr;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = buildFlags_;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = as_;
    buildInfo.dstAccelerationStructure = as_;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;

    VkDeviceAddress scratchAddr = scratch.deviceAddress();
    if (scratchAlignment_ > 1) {
        scratchAddr = alignUp(scratchAddr, scratchAlignment_);
    }
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;

    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &rangeInfo;
    fn.cmdBuildAs(cmd, 1, &buildInfo, &pRanges);

    return {};
}

} // namespace vksdl
