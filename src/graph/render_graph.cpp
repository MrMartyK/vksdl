#include <vksdl/allocator.hpp>
#include <vksdl/buffer.hpp>
#include <vksdl/descriptor_allocator.hpp>
#include <vksdl/descriptor_writer.hpp>
#include <vksdl/device.hpp>
#include <vksdl/graph/render_graph.hpp>
#include <vksdl/image.hpp>
#include <vksdl/shader_reflect.hpp>

#include <vk_mem_alloc.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <queue>

namespace vksdl::graph {

// Cast void* back to VmaAllocator for internal use.
static VmaAllocator toVma(void* p) {
    return static_cast<VmaAllocator>(p);
}

RenderGraph::RenderGraph(const Device& device, const Allocator& allocator)
    : device_(device.vkDevice()), allocator_(allocator.vmaAllocator()),
      hasUnifiedLayouts_(device.hasUnifiedImageLayouts()) {
    auto alloc = DescriptorAllocator::create(device);
    if (alloc.ok()) {
        descAllocator_ = std::make_unique<DescriptorAllocator>(std::move(alloc).value());
    } else {
#ifndef NDEBUG
        std::fprintf(stderr,
                     "[vksdl::graph] DescriptorAllocator creation failed: %s\n"
                     "  Layer 2 auto-bind will be unavailable.\n",
                     alloc.error().message.c_str());
#endif
    }
}

RenderGraph::~RenderGraph() {
    destroy();
}

RenderGraph::RenderGraph(RenderGraph&& o) noexcept
    : device_(o.device_), allocator_(o.allocator_), hasUnifiedLayouts_(o.hasUnifiedLayouts_),
      passes_(std::move(o.passes_)), resources_(std::move(o.resources_)),
      imageMaps_(std::move(o.imageMaps_)), bufferStates_(std::move(o.bufferStates_)),
      adj_(std::move(o.adj_)), inDegree_(std::move(o.inDegree_)),
      compiledPasses_(std::move(o.compiledPasses_)), isCompiled_(o.isCompiled_),
      stats_(std::move(o.stats_)), transientImages_(std::move(o.transientImages_)),
      transientBuffers_(std::move(o.transientBuffers_)), imagePool_(std::move(o.imagePool_)),
      bufferPool_(std::move(o.bufferPool_)), lastGraphHash_(o.lastGraphHash_),
      cachedOrder_(std::move(o.cachedOrder_)),
      cachedImageHandles_(std::move(o.cachedImageHandles_)),
      cachedViewHandles_(std::move(o.cachedViewHandles_)),
      cachedBufferHandles_(std::move(o.cachedBufferHandles_)),
      cachedStats_(std::move(o.cachedStats_)), descAllocator_(std::move(o.descAllocator_)),
      dslCache_(std::move(o.dslCache_)) {
    o.device_ = VK_NULL_HANDLE;
    o.allocator_ = nullptr;
    o.isCompiled_ = false;
    o.lastGraphHash_ = 0;
}

RenderGraph& RenderGraph::operator=(RenderGraph&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_;
        allocator_ = o.allocator_;
        hasUnifiedLayouts_ = o.hasUnifiedLayouts_;
        passes_ = std::move(o.passes_);
        resources_ = std::move(o.resources_);
        imageMaps_ = std::move(o.imageMaps_);
        bufferStates_ = std::move(o.bufferStates_);
        adj_ = std::move(o.adj_);
        inDegree_ = std::move(o.inDegree_);
        compiledPasses_ = std::move(o.compiledPasses_);
        isCompiled_ = o.isCompiled_;
        stats_ = std::move(o.stats_);
        transientImages_ = std::move(o.transientImages_);
        transientBuffers_ = std::move(o.transientBuffers_);
        imagePool_ = std::move(o.imagePool_);
        bufferPool_ = std::move(o.bufferPool_);
        lastGraphHash_ = o.lastGraphHash_;
        cachedOrder_ = std::move(o.cachedOrder_);
        cachedImageHandles_ = std::move(o.cachedImageHandles_);
        cachedViewHandles_ = std::move(o.cachedViewHandles_);
        cachedBufferHandles_ = std::move(o.cachedBufferHandles_);
        cachedStats_ = std::move(o.cachedStats_);
        descAllocator_ = std::move(o.descAllocator_);
        dslCache_ = std::move(o.dslCache_);
        o.device_ = VK_NULL_HANDLE;
        o.allocator_ = nullptr;
        o.isCompiled_ = false;
        o.lastGraphHash_ = 0;
    }
    return *this;
}

void RenderGraph::destroy() {
    // Layer 2: destroy cached descriptor set layouts.
    for (auto dsl : dslCache_)
        if (dsl != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
    dslCache_.clear();

    descAllocator_.reset();

    destroyTransients();
    destroyPool();
}

void RenderGraph::destroyTransients() {
    if (device_ == VK_NULL_HANDLE)
        return;
    auto vma = toVma(allocator_);

    for (auto& t : transientImages_) {
        if (t.view != VK_NULL_HANDLE)
            vkDestroyImageView(device_, t.view, nullptr);
        if (t.image != VK_NULL_HANDLE)
            vmaDestroyImage(vma, t.image, static_cast<VmaAllocation>(t.allocation));
    }
    transientImages_.clear();

    for (auto& t : transientBuffers_) {
        if (t.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(vma, t.buffer, static_cast<VmaAllocation>(t.allocation));
    }
    transientBuffers_.clear();
}

void RenderGraph::recycleTransients() {
    // Move active transients to pool for reuse next frame. No VMA calls.
    for (auto& t : transientImages_)
        imagePool_.push_back(std::move(t));
    transientImages_.clear();

    for (auto& t : transientBuffers_)
        bufferPool_.push_back(std::move(t));
    transientBuffers_.clear();
}

void RenderGraph::destroyPool() {
    if (device_ == VK_NULL_HANDLE)
        return;
    auto vma = toVma(allocator_);

    for (auto& t : imagePool_) {
        if (t.view != VK_NULL_HANDLE)
            vkDestroyImageView(device_, t.view, nullptr);
        if (t.image != VK_NULL_HANDLE)
            vmaDestroyImage(vma, t.image, static_cast<VmaAllocation>(t.allocation));
    }
    imagePool_.clear();

    for (auto& t : bufferPool_) {
        if (t.buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(vma, t.buffer, static_cast<VmaAllocation>(t.allocation));
    }
    bufferPool_.clear();
}

ResourceHandle RenderGraph::importImage(VkImage image, VkImageView view, VkFormat format,
                                        std::uint32_t width, std::uint32_t height,
                                        const ResourceState& initialState, std::uint32_t mipLevels,
                                        std::uint32_t arrayLayers, std::string_view name) {

    ResourceHandle h{static_cast<std::uint32_t>(resources_.size())};

    ResourceEntry entry{};
    entry.tag = ResourceTag::External;
    entry.kind = ResourceKind::Image;
    entry.name = name;
    entry.vkImage = image;
    entry.vkImageView = view;
    entry.imageDesc = {width, height, format, 0, mipLevels, arrayLayers, VK_SAMPLE_COUNT_1_BIT};
    entry.aspect = aspectFromFormat(format);
    entry.initialState = initialState;
    resources_.push_back(entry);

    return h;
}

ResourceHandle RenderGraph::importImage(const Image& image, const ResourceState& initialState,
                                        std::string_view name) {
    return importImage(image.vkImage(), image.vkImageView(), image.format(), image.extent().width,
                       image.extent().height, initialState, image.mipLevels(), 1, name);
}

ResourceHandle RenderGraph::importBuffer(VkBuffer buffer, VkDeviceSize size,
                                         const ResourceState& initialState, std::string_view name) {

    ResourceHandle h{static_cast<std::uint32_t>(resources_.size())};

    ResourceEntry entry{};
    entry.tag = ResourceTag::External;
    entry.kind = ResourceKind::Buffer;
    entry.name = name;
    entry.vkBuffer = buffer;
    entry.bufferSize = size;
    entry.initialState = initialState;
    resources_.push_back(entry);

    return h;
}

ResourceHandle RenderGraph::importBuffer(const Buffer& buffer, const ResourceState& initialState,
                                         std::string_view name) {
    return importBuffer(buffer.vkBuffer(), buffer.size(), initialState, name);
}

ResourceHandle RenderGraph::createImage(const ImageDesc& desc, std::string_view name) {
    ResourceHandle h{static_cast<std::uint32_t>(resources_.size())};

    ResourceEntry entry{};
    entry.tag = ResourceTag::Transient;
    entry.kind = ResourceKind::Image;
    entry.name = name;
    entry.imageDesc = desc;
    entry.aspect = aspectFromFormat(desc.format);
    resources_.push_back(entry);

    return h;
}

ResourceHandle RenderGraph::createBuffer(const BufferDesc& desc, std::string_view name) {
    ResourceHandle h{static_cast<std::uint32_t>(resources_.size())};

    ResourceEntry entry{};
    entry.tag = ResourceTag::Transient;
    entry.kind = ResourceKind::Buffer;
    entry.name = name;
    entry.bufferDesc = desc;
    resources_.push_back(entry);

    return h;
}

void RenderGraph::addPass(std::string_view name, PassType type, SetupFn setup, RecordFn record) {
    PassBuilder builder(type);
    setup(builder);

    PassDecl decl;
    decl.name = std::string(name);
    decl.type = type;
    decl.accesses = std::move(builder.accesses_);
    decl.recordFn = std::move(record);
    decl.colorTargets = std::move(builder.colorTargets_);
    decl.depthTarget = std::move(builder.depthTarget_);
    passes_.push_back(std::move(decl));
}

void RenderGraph::addPass(std::string_view name, PassType type, VkPipeline pipeline,
                          VkPipelineLayout pipelineLayout, const ReflectedLayout& reflection,
                          SetupFn setup, RecordFn record) {
    PassBuilder builder(type);
    setup(builder);

    // Access inference: for each reflected binding that has a bind map entry,
    // call the appropriate Layer 0 method so the barrier compiler sees it.
    for (const auto& rb : reflection.bindings) {
        auto it = builder.bindMap_.find(rb.name);
        if (it == builder.bindMap_.end())
            continue;

        ResourceHandle h = it->second.handle;
        switch (rb.type) {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            builder.sampleImage(h);
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            // Read-only default. For writes, user calls writeStorageImage()
            // in the setup lambda alongside bind().
            builder.readStorageImage(h);
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            builder.readUniformBuffer(h);
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            // Read-only default. For writes, user calls writeStorageBuffer()
            // in the setup lambda alongside bind().
            builder.readStorageBuffer(h);
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            builder.readInputAttachment(h);
            break;
        default:
            break;
        }
    }

    PassDecl decl;
    decl.name = std::string(name);
    decl.type = type;
    decl.accesses = std::move(builder.accesses_);
    decl.recordFn = std::move(record);
    decl.colorTargets = std::move(builder.colorTargets_);
    decl.depthTarget = std::move(builder.depthTarget_);
    decl.pipeline = pipeline;
    decl.pipelineLayout = pipelineLayout;
    decl.reflection = &reflection;
    decl.defaultSampler = builder.defaultSampler_;
    decl.bindMap = std::move(builder.bindMap_);
    passes_.push_back(std::move(decl));
}

void RenderGraph::resolveRemainingCounts() {
    for (auto& pass : passes_) {
        for (auto& acc : pass.accesses) {
            if (!acc.handle.valid())
                continue;
            const auto& res = resources_[acc.handle.index];
            if (res.kind != ResourceKind::Image)
                continue;

            auto& r = acc.subresourceRange;
            if (r.levelCount == VK_REMAINING_MIP_LEVELS)
                r.levelCount = res.imageDesc.mipLevels - r.baseMipLevel;
            if (r.layerCount == VK_REMAINING_ARRAY_LAYERS)
                r.layerCount = res.imageDesc.arrayLayers - r.baseArrayLayer;
        }
    }
}

void RenderGraph::accumulateTransientUsage() {
    for (const auto& pass : passes_) {
        for (const auto& acc : pass.accesses) {
            if (!acc.handle.valid())
                continue;
            auto& res = resources_[acc.handle.index];
            if (res.tag != ResourceTag::Transient)
                continue;

            if (res.kind == ResourceKind::Image) {
                VkImageLayout layout = acc.desiredState.currentLayout;

                if (layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    res.imageDesc.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                else if (layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    res.imageDesc.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                else if (layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                    res.imageDesc.usage |=
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                else if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    res.imageDesc.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                else if (layout == VK_IMAGE_LAYOUT_GENERAL) {
                    if (isWriteAccess(acc.desiredState.lastWriteAccess) ||
                        (acc.desiredState.readAccessSinceWrite &
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT))
                        res.imageDesc.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                } else if (layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                    res.imageDesc.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                else if (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    res.imageDesc.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

                // Input attachment.
                if (acc.desiredState.readAccessSinceWrite & VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT)
                    res.imageDesc.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

            } else {
                // Buffer: accumulate from access masks.
                VkAccessFlags2 a =
                    acc.desiredState.lastWriteAccess | acc.desiredState.readAccessSinceWrite;

                if (a & VK_ACCESS_2_UNIFORM_READ_BIT)
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                if (a &
                    (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT))
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                if (a & VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
                if (a & VK_ACCESS_2_INDEX_READ_BIT)
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
                if (a & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT)
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
                if (a & VK_ACCESS_2_TRANSFER_READ_BIT)
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                if (a & VK_ACCESS_2_TRANSFER_WRITE_BIT)
                    res.bufferDesc.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            }
        }
    }
}

void RenderGraph::buildAdjacency() {
    const auto passCount = static_cast<std::uint32_t>(passes_.size());
    const auto resCount = static_cast<std::uint32_t>(resources_.size());

    // Flat bool matrix for deduplication (avoids hash map allocation).
    std::vector<bool> adjMatrix(static_cast<std::size_t>(passCount) * passCount, false);

    // Collect per-resource writers and readers using flat arrays.
    // Each resource gets a small inline list of pass indices.
    // For typical graphs (6 passes, 7 resources), this is ~50 uint32s total.
    struct ResAccess {
        std::vector<std::uint32_t> writers;
        std::vector<std::uint32_t> readers;
    };
    std::vector<ResAccess> perResource(resCount);

    for (std::uint32_t pi = 0; pi < passCount; ++pi) {
        for (const auto& acc : passes_[pi].accesses) {
            if (!acc.handle.valid())
                continue;
            std::uint32_t ri = acc.handle.index;
            auto& ra = perResource[ri];

            bool writes = (acc.access == AccessType::Write || acc.access == AccessType::ReadWrite);
            bool reads = (acc.access == AccessType::Read || acc.access == AccessType::ReadWrite);

            if (writes)
                ra.writers.push_back(pi);
            if (reads)
                ra.readers.push_back(pi);
        }
    }

    // Generate edges from resource data-flow. All edges point forward in
    // declaration order (lower pass index -> higher), so ping-pong and
    // multi-substep patterns never produce false cycles.
    //
    // RAW (w < r): writer declared before reader -- reader depends on writer.
    // WAR (r < w): reader declared before writer -- reader must finish before
    //              the writer overwrites the resource.
    // WAW: consecutive writers in declaration order.
    for (std::uint32_t ri = 0; ri < resCount; ++ri) {
        const auto& ra = perResource[ri];

        for (auto w : ra.writers) {
            for (auto r : ra.readers) {
                if (w < r)
                    adjMatrix[w * passCount + r] = true; // RAW
                else if (r < w)
                    adjMatrix[r * passCount + w] = true; // WAR
            }
        }

        for (std::size_t wi = 1; wi < ra.writers.size(); ++wi)
            adjMatrix[ra.writers[wi - 1] * passCount + ra.writers[wi]] = true;
    }

    // Convert bool matrix to adjacency list + in-degree.
    adj_.assign(passCount, {});
    inDegree_.assign(passCount, 0);

    for (std::uint32_t i = 0; i < passCount; ++i) {
        for (std::uint32_t j = 0; j < passCount; ++j) {
            if (adjMatrix[i * passCount + j]) {
                adj_[i].push_back(j);
                inDegree_[j]++;
            }
        }
    }
}

Result<std::vector<std::uint32_t>> RenderGraph::topologicalSort() {
    const auto passCount = static_cast<std::uint32_t>(passes_.size());

    // Work on a copy of in-degree since Kahn's mutates it.
    auto inDeg = inDegree_;

    std::queue<std::uint32_t> q;
    for (std::uint32_t i = 0; i < passCount; ++i) {
        if (inDeg[i] == 0)
            q.push(i);
    }

    std::vector<std::uint32_t> order;
    order.reserve(passCount);

    while (!q.empty()) {
        auto u = q.front();
        q.pop();
        order.push_back(u);
        for (auto v : adj_[u]) {
            if (--inDeg[v] == 0)
                q.push(v);
        }
    }

    if (order.size() != passCount) {
        return Error{"compile render graph", 0, "cycle detected in pass dependencies"};
    }

    return order;
}

void RenderGraph::computeLifetimes(const std::vector<std::uint32_t>& order) {
    // Map from original pass index to sorted position.
    std::vector<std::uint32_t> sortedPos(passes_.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(order.size()); ++i)
        sortedPos[order[i]] = i;

    for (std::uint32_t pi = 0; pi < static_cast<std::uint32_t>(passes_.size()); ++pi) {
        for (const auto& acc : passes_[pi].accesses) {
            if (!acc.handle.valid())
                continue;
            auto& res = resources_[acc.handle.index];
            std::uint32_t pos = sortedPos[pi];
            if (pos < res.firstPass)
                res.firstPass = pos;
            if (pos > res.lastPass || res.lastPass == UINT32_MAX)
                res.lastPass = pos;
        }
    }
}

Result<void> RenderGraph::allocateTransients() {
    auto vma = toVma(allocator_);

    // Fast path: when pool sizes match transient count exactly (steady-state
    // after prewarm or second frame), consume pool entries sequentially.
    // recycleTransients() preserves insertion order, so pool[i] maps to the
    // i-th transient image/buffer resource. This avoids linear scan and
    // std::vector<bool> allocation entirely.
    std::uint32_t imgPoolIdx = 0;
    std::uint32_t bufPoolIdx = 0;

    // Count transient images/buffers for fast-path eligibility.
    std::uint32_t transImgCount = 0, transBufCount = 0;
    for (const auto& res : resources_) {
        if (res.tag != ResourceTag::Transient)
            continue;
        if (res.kind == ResourceKind::Image)
            ++transImgCount;
        else
            ++transBufCount;
    }
    bool fastPool = (imagePool_.size() == transImgCount && bufferPool_.size() == transBufCount);

    for (std::uint32_t ri = 0; ri < static_cast<std::uint32_t>(resources_.size()); ++ri) {
        auto& res = resources_[ri];
        if (res.tag != ResourceTag::Transient)
            continue;

        if (res.kind == ResourceKind::Image) {
            bool found = false;

            if (fastPool && imgPoolIdx < static_cast<std::uint32_t>(imagePool_.size())) {
                // Sequential pool consumption (deterministic, no search).
                auto& pooled = imagePool_[imgPoolIdx++];
                res.vkImage = pooled.image;
                res.vkImageView = pooled.view;
                transientImages_.push_back(pooled);
                found = true;
            } else if (!fastPool) {
                // Slow path: linear scan with mark-used tracking.
                for (std::size_t pi = 0; pi < imagePool_.size(); ++pi) {
                    if (imagePool_[pi].desc == res.imageDesc &&
                        imagePool_[pi].image != VK_NULL_HANDLE) {
                        res.vkImage = imagePool_[pi].image;
                        res.vkImageView = imagePool_[pi].view;
                        transientImages_.push_back(imagePool_[pi]);
                        imagePool_[pi].image = VK_NULL_HANDLE; // mark consumed
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                VkImageCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ci.imageType = VK_IMAGE_TYPE_2D;
                ci.format = res.imageDesc.format;
                ci.extent = {res.imageDesc.width, res.imageDesc.height, 1};
                ci.mipLevels = res.imageDesc.mipLevels;
                ci.arrayLayers = res.imageDesc.arrayLayers;
                ci.samples = res.imageDesc.samples;
                ci.tiling = VK_IMAGE_TILING_OPTIMAL;
                ci.usage = res.imageDesc.usage;
                ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                VmaAllocationCreateInfo allocCI{};
                allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;

                VkImage image = VK_NULL_HANDLE;
                VmaAllocation allocation = nullptr;
                VkResult vr = vmaCreateImage(vma, &ci, &allocCI, &image, &allocation, nullptr);
                if (vr != VK_SUCCESS) {
                    return Error{"allocate transient image", vr,
                                 "VMA failed to allocate transient image"};
                }

                VkImageViewCreateInfo viewCI{};
                viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewCI.image = image;
                viewCI.viewType = (res.imageDesc.arrayLayers > 1) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                  : VK_IMAGE_VIEW_TYPE_2D;
                viewCI.format = res.imageDesc.format;
                viewCI.subresourceRange = {
                    res.aspect, 0, res.imageDesc.mipLevels, 0, res.imageDesc.arrayLayers,
                };

                VkImageView view = VK_NULL_HANDLE;
                vr = vkCreateImageView(device_, &viewCI, nullptr, &view);
                if (vr != VK_SUCCESS) {
                    vmaDestroyImage(vma, image, allocation);
                    return Error{"create transient image view", vr,
                                 "failed to create image view for transient"};
                }

                res.vkImage = image;
                res.vkImageView = view;
                transientImages_.push_back({res.imageDesc, image, view, allocation});
            }

        } else {
            bool found = false;

            if (fastPool && bufPoolIdx < static_cast<std::uint32_t>(bufferPool_.size())) {
                auto& pooled = bufferPool_[bufPoolIdx++];
                res.vkBuffer = pooled.buffer;
                res.bufferSize = res.bufferDesc.size;
                transientBuffers_.push_back(pooled);
                found = true;
            } else if (!fastPool) {
                for (std::size_t pi = 0; pi < bufferPool_.size(); ++pi) {
                    if (bufferPool_[pi].desc == res.bufferDesc &&
                        bufferPool_[pi].buffer != VK_NULL_HANDLE) {
                        res.vkBuffer = bufferPool_[pi].buffer;
                        res.bufferSize = res.bufferDesc.size;
                        transientBuffers_.push_back(bufferPool_[pi]);
                        bufferPool_[pi].buffer = VK_NULL_HANDLE; // mark consumed
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                VkBufferCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                ci.size = res.bufferDesc.size;
                ci.usage = res.bufferDesc.usage;
                ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VmaAllocationCreateInfo allocCI{};
                allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;

                VkBuffer buffer = VK_NULL_HANDLE;
                VmaAllocation allocation = nullptr;
                VkResult vr = vmaCreateBuffer(vma, &ci, &allocCI, &buffer, &allocation, nullptr);
                if (vr != VK_SUCCESS) {
                    return Error{"allocate transient buffer", vr,
                                 "VMA failed to allocate transient buffer"};
                }

                res.vkBuffer = buffer;
                res.bufferSize = res.bufferDesc.size;
                transientBuffers_.push_back({res.bufferDesc, buffer, allocation});
            }
        }
    }

    // Destroy unclaimed pool entries (e.g., stale after resize).
    if (fastPool) {
        // All entries consumed sequentially.
        imagePool_.clear();
        bufferPool_.clear();
    } else {
        // Remove consumed entries (marked with VK_NULL_HANDLE).
        std::erase_if(imagePool_,
                      [](const TransientImage& t) { return t.image == VK_NULL_HANDLE; });
        std::erase_if(bufferPool_,
                      [](const TransientBuffer& t) { return t.buffer == VK_NULL_HANDLE; });
        destroyPool();
    }

    return {};
}

void RenderGraph::initStateTrackers() {
    imageMaps_.clear();
    bufferStates_.clear();

    // Resize to match resources_. imageMaps_ and bufferStates_ are indexed
    // by resource index. Only the relevant kind is meaningful.
    imageMaps_.reserve(resources_.size());
    bufferStates_.resize(resources_.size());

    for (std::uint32_t ri = 0; ri < static_cast<std::uint32_t>(resources_.size()); ++ri) {
        const auto& res = resources_[ri];
        if (res.kind == ResourceKind::Image) {
            imageMaps_.emplace_back(res.imageDesc.mipLevels, res.imageDesc.arrayLayers,
                                    res.initialState);
        } else {
            imageMaps_.emplace_back(1, 1); // placeholder
            bufferStates_[ri] = res.initialState;
        }
    }
}

static Result<void> validateQueueFamilyTransition(const ResourceState& src,
                                                  const ResourceState& dst,
                                                  const ResourceEntry& res) {
    if (src.queueFamily == VK_QUEUE_FAMILY_IGNORED || dst.queueFamily == VK_QUEUE_FAMILY_IGNORED ||
        src.queueFamily == dst.queueFamily) {
        return {};
    }

    std::string resourceName = res.name.empty() ? "(unnamed)" : res.name;
    return Error{"compile render graph", 0,
                 "queue-family ownership transfer requested for resource '" + resourceName +
                     "' (src=" + std::to_string(src.queueFamily) +
                     ", dst=" + std::to_string(dst.queueFamily) +
                     "). RenderGraph executes on one queue family; explicit release/acquire"
                     " ownership transfer (including maintenance9 caveats) is not modeled yet."};
}

Result<void> RenderGraph::compileBarriers(const std::vector<std::uint32_t>& order) {
    compiledPasses_.clear();
    compiledPasses_.reserve(order.size());

    for (auto passIdx : order) {
        CompiledPass cp;
        cp.passIndex = passIdx;
        const auto& pass = passes_[passIdx];

        for (const auto& acc : pass.accesses) {
            if (!acc.handle.valid())
                continue;
            std::uint32_t ri = acc.handle.index;
            const auto& res = resources_[ri];

            bool isRead = (acc.access == AccessType::Read);

            if (res.kind == ResourceKind::Image) {
                // Walk actual slices from the ImageSubresourceMap that overlap
                // this access's subresource range.
                auto& map = imageMaps_[ri];
                const auto& slices = map.slices();

                for (const auto& slice : slices) {
                    if (!slice.range.overlaps(acc.subresourceRange))
                        continue;

                    auto queueCheck =
                        validateQueueFamilyTransition(slice.state, acc.desiredState, res);
                    if (!queueCheck) {
                        return queueCheck.error();
                    }

                    // Compute the overlap region.
                    SubresourceRange overlap;
                    overlap.baseMipLevel =
                        std::max(slice.range.baseMipLevel, acc.subresourceRange.baseMipLevel);
                    overlap.baseArrayLayer =
                        std::max(slice.range.baseArrayLayer, acc.subresourceRange.baseArrayLayer);
                    std::uint32_t mipEnd =
                        std::min(slice.range.mipEnd(), acc.subresourceRange.mipEnd());
                    std::uint32_t layerEnd =
                        std::min(slice.range.layerEnd(), acc.subresourceRange.layerEnd());
                    overlap.levelCount = mipEnd - overlap.baseMipLevel;
                    overlap.layerCount = layerEnd - overlap.baseArrayLayer;

                    // When unified layouts are active, suppress layout
                    // transitions between known layouts. The initial
                    // UNDEFINED -> X transition is preserved (spec requires it).
                    ResourceState srcState = slice.state;
                    ResourceState dstState = acc.desiredState;
                    if (hasUnifiedLayouts_ && srcState.currentLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
                        srcState.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                        dstState.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                    }

                    appendImageBarrier(cp.barriers, ImageBarrierRequest{
                                                        .image = res.vkImage,
                                                        .range = overlap,
                                                        .aspect = res.aspect,
                                                        .src = srcState,
                                                        .dst = dstState,
                                                        .isRead = isRead,
                                                    });
                }

                // Update tracked state for the accessed range.
                ResourceState newState = acc.desiredState;
                if (hasUnifiedLayouts_) {
                    newState.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                }
                if (isRead) {
                    // Preserve writer info from current state, add this read.
                    ResourceState merged = map.queryState(acc.subresourceRange);
                    newState.lastWriteStage = merged.lastWriteStage;
                    newState.lastWriteAccess = merged.lastWriteAccess;
                    newState.readStagesSinceWrite =
                        merged.readStagesSinceWrite | acc.desiredState.lastWriteStage;
                    newState.readAccessSinceWrite =
                        merged.readAccessSinceWrite | acc.desiredState.readAccessSinceWrite;
                    newState.currentLayout = acc.desiredState.currentLayout;
                    if (newState.queueFamily == VK_QUEUE_FAMILY_IGNORED) {
                        newState.queueFamily = merged.queueFamily;
                    }
                } else {
                    // Write: reset readers, set new writer.
                    newState.readStagesSinceWrite = VK_PIPELINE_STAGE_2_NONE;
                    newState.readAccessSinceWrite = VK_ACCESS_2_NONE;
                    if (newState.queueFamily == VK_QUEUE_FAMILY_IGNORED) {
                        newState.queueFamily = map.queryState(acc.subresourceRange).queueFamily;
                    }
                }
                map.setState(acc.subresourceRange, newState);

            } else {
                // Buffer: single state, no subresources.
                auto& state = bufferStates_[ri];

                auto queueCheck = validateQueueFamilyTransition(state, acc.desiredState, res);
                if (!queueCheck) {
                    return queueCheck.error();
                }

                appendBufferBarrier(cp.barriers, BufferBarrierRequest{
                                                     .buffer = res.vkBuffer,
                                                     .offset = 0,
                                                     .size = VK_WHOLE_SIZE,
                                                     .src = state,
                                                     .dst = acc.desiredState,
                                                     .isRead = isRead,
                                                 });

                // Update tracked state.
                if (isRead) {
                    state.readStagesSinceWrite |= acc.desiredState.lastWriteStage;
                    state.readAccessSinceWrite |= acc.desiredState.readAccessSinceWrite;
                    if (acc.desiredState.queueFamily != VK_QUEUE_FAMILY_IGNORED) {
                        state.queueFamily = acc.desiredState.queueFamily;
                    }
                } else {
                    std::uint32_t previousQueueFamily = state.queueFamily;
                    state = acc.desiredState;
                    state.readStagesSinceWrite = VK_PIPELINE_STAGE_2_NONE;
                    state.readAccessSinceWrite = VK_ACCESS_2_NONE;
                    if (state.queueFamily == VK_QUEUE_FAMILY_IGNORED) {
                        state.queueFamily = previousQueueFamily;
                    }
                }
            }
        }

        compiledPasses_.push_back(std::move(cp));
    }

    return {};
}

static VkAttachmentLoadOp toVkLoadOp(LoadOp op) {
    switch (op) {
    case LoadOp::Clear:
        return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::Load:
        return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::DontCare:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE; // unreachable
}

void RenderGraph::resolveRenderTargets(const std::vector<std::uint32_t>& order) {
    // Build sorted position map (pass index -> position in execution order).
    std::vector<std::uint32_t> sortedPos(passes_.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(order.size()); ++i)
        sortedPos[order[i]] = i;

    for (auto& cp : compiledPasses_) {
        const auto& passDecl = passes_[cp.passIndex];

        // Skip passes with no Layer 1 targets (pure Layer 0 usage).
        if (passDecl.colorTargets.empty() && !passDecl.depthTarget)
            continue;

        auto& rr = cp.rendering;
        std::uint32_t myPos = sortedPos[cp.passIndex];

        // Derive render area from the first declared target.
        ResourceHandle areaHandle;
        if (!passDecl.colorTargets.empty())
            areaHandle = passDecl.colorTargets[0].handle;
        else if (passDecl.depthTarget)
            areaHandle = passDecl.depthTarget->handle;

        if (areaHandle.valid()) {
            const auto& res = resources_[areaHandle.index];
            rr.renderArea = {res.imageDesc.width, res.imageDesc.height};
        }

        // Resolve color attachments.
        // Determine the max color attachment index to size the vector.
        std::uint32_t maxIndex = 0;
        for (const auto& ct : passDecl.colorTargets)
            if (ct.index >= maxIndex)
                maxIndex = ct.index + 1;

        rr.colorAttachments.resize(maxIndex);
        for (auto& att : rr.colorAttachments) {
            att = {};
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = VK_NULL_HANDLE;
        }

        for (const auto& ct : passDecl.colorTargets) {
            const auto& res = resources_[ct.handle.index];

            auto& att = rr.colorAttachments[ct.index];
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = res.vkImageView;
            att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att.loadOp = toVkLoadOp(ct.loadOp);
            att.clearValue.color = ct.clearValue;

            // Store op inference: transient last use -> DONT_CARE, else STORE.
            if (res.tag == ResourceTag::Transient && myPos >= res.lastPass)
                att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            else
                att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }

        // Resolve depth attachment.
        if (passDecl.depthTarget) {
            const auto& dt = *passDecl.depthTarget;
            const auto& res = resources_[dt.handle.index];

            rr.hasDepth = true;
            auto& att = rr.depthAttachment;
            att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            att.imageView = res.vkImageView;
            att.loadOp = toVkLoadOp(dt.loadOp);
            att.clearValue.depthStencil = {dt.clearDepth, dt.clearStencil};

            if (dt.depthWrite == DepthWrite::Enabled) {
                att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                // Store op: transient last use -> DONT_CARE, else STORE.
                if (res.tag == ResourceTag::Transient && myPos >= res.lastPass)
                    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                else
                    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            } else {
                att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                att.storeOp = VK_ATTACHMENT_STORE_OP_NONE;
            }
        }
    }
}

Result<void> RenderGraph::resolveDescriptors() {
    if (!descAllocator_)
        return {};

    for (auto& cp : compiledPasses_) {
        const auto& passDecl = passes_[cp.passIndex];
        if (!passDecl.reflection)
            continue;

        auto& desc = cp.descriptors;
        desc.pipeline = passDecl.pipeline;
        desc.pipelineLayout = passDecl.pipelineLayout;
        desc.bindPoint = (passDecl.type == PassType::Compute) ? VK_PIPELINE_BIND_POINT_COMPUTE
                                                              : VK_PIPELINE_BIND_POINT_GRAPHICS;

        // Find max set index.
        std::uint32_t maxSet = 0;
        for (const auto& rb : passDecl.reflection->bindings)
            if (rb.set >= maxSet)
                maxSet = rb.set + 1;

        desc.sets.assign(maxSet, VK_NULL_HANDLE);

        // Process each set.
        for (std::uint32_t si = 0; si < maxSet; ++si) {
            // Collect bindings for this set.
            std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
            bool hasGraphManaged = false;

            for (const auto& rb : passDecl.reflection->bindings) {
                if (rb.set != si)
                    continue;

                VkDescriptorSetLayoutBinding lb{};
                lb.binding = rb.binding;
                lb.descriptorType = rb.type;
                lb.descriptorCount = rb.count;
                lb.stageFlags = rb.stages;
                layoutBindings.push_back(lb);

                if (passDecl.bindMap.count(rb.name))
                    hasGraphManaged = true;
            }

            if (!hasGraphManaged || layoutBindings.empty())
                continue;

            // Create DSL.
            VkDescriptorSetLayoutCreateInfo dslCI{};
            dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslCI.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
            dslCI.pBindings = layoutBindings.data();

            VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
            VkResult dslResult = vkCreateDescriptorSetLayout(device_, &dslCI, nullptr, &dsl);
            if (dslResult != VK_SUCCESS)
                return Error{"vkCreateDescriptorSetLayout", static_cast<int32_t>(dslResult),
                             "failed to create descriptor set layout for pass '" +
                                 std::string(passDecl.name) + "' set " + std::to_string(si)};
            dslCache_.push_back(dsl);

            // Allocate set.
            auto setResult = descAllocator_->allocate(dsl);
            if (!setResult.ok())
                return Error{"DescriptorAllocator::allocate", 0,
                             "failed to allocate descriptor set for pass '" +
                                 std::string(passDecl.name) + "' set " + std::to_string(si)};
            VkDescriptorSet set = setResult.value();
            desc.sets[si] = set;

            // Write descriptors.
            DescriptorWriter writer(set);

            for (const auto& rb : passDecl.reflection->bindings) {
                if (rb.set != si)
                    continue;
                auto it = passDecl.bindMap.find(rb.name);
                if (it == passDecl.bindMap.end())
                    continue;

                ResourceHandle h = it->second.handle;
                if (!h.valid())
                    continue;
                const auto& res = resources_[h.index];

                switch (rb.type) {
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: {
                    VkSampler sampler = it->second.samplerOverride != VK_NULL_HANDLE
                                            ? it->second.samplerOverride
                                            : passDecl.defaultSampler;
                    writer.image(rb.binding, res.vkImageView,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sampler, rb.type);
                    break;
                }
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    writer.storageImage(rb.binding, res.vkImageView, VK_IMAGE_LAYOUT_GENERAL);
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    writer.buffer(rb.binding, res.vkBuffer, res.bufferSize, 0, rb.type);
                    break;
                default:
                    break;
                }
            }

            writer.write(device_);
        }
    }
    return {};
}

// FNV-1a hash for graph structure caching.
static std::uint64_t fnv1a(const void* data, std::size_t len,
                           std::uint64_t h = 14695981039346656037ULL) {
    auto p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static std::uint64_t hashGraphStructure(const std::vector<PassDecl>& passes,
                                        const std::vector<ResourceEntry>& resources) {
    // Hash pass count + resource count.
    auto passCount = static_cast<std::uint32_t>(passes.size());
    auto resCount = static_cast<std::uint32_t>(resources.size());
    auto h = fnv1a(&passCount, sizeof(passCount));
    h = fnv1a(&resCount, sizeof(resCount), h);

    // Hash each pass: type + access list + render target metadata.
    for (const auto& pass : passes) {
        h = fnv1a(&pass.type, sizeof(pass.type), h);
        auto accCount = static_cast<std::uint32_t>(pass.accesses.size());
        h = fnv1a(&accCount, sizeof(accCount), h);
        for (const auto& acc : pass.accesses) {
            h = fnv1a(&acc.handle.index, sizeof(acc.handle.index), h);
            h = fnv1a(&acc.access, sizeof(acc.access), h);
            h = fnv1a(&acc.desiredState.currentLayout, sizeof(acc.desiredState.currentLayout), h);
            h = fnv1a(&acc.subresourceRange, sizeof(acc.subresourceRange), h);
        }
        // Layer 1: color target count + indices + loadOps.
        auto ctCount = static_cast<std::uint32_t>(pass.colorTargets.size());
        h = fnv1a(&ctCount, sizeof(ctCount), h);
        for (const auto& ct : pass.colorTargets) {
            h = fnv1a(&ct.index, sizeof(ct.index), h);
            h = fnv1a(&ct.loadOp, sizeof(ct.loadOp), h);
            h = fnv1a(&ct.handle.index, sizeof(ct.handle.index), h);
        }
        // Layer 1: depth target presence + loadOp + depthWrite.
        bool hasDT = pass.depthTarget.has_value();
        h = fnv1a(&hasDT, sizeof(hasDT), h);
        if (hasDT) {
            h = fnv1a(&pass.depthTarget->loadOp, sizeof(pass.depthTarget->loadOp), h);
            h = fnv1a(&pass.depthTarget->depthWrite, sizeof(pass.depthTarget->depthWrite), h);
            h = fnv1a(&pass.depthTarget->handle.index, sizeof(pass.depthTarget->handle.index), h);
        }
        // Layer 2: pipeline + reflection pointer + default sampler + bind count.
        h = fnv1a(&pass.pipeline, sizeof(pass.pipeline), h);
        h = fnv1a(&pass.pipelineLayout, sizeof(pass.pipelineLayout), h);
        h = fnv1a(&pass.reflection, sizeof(pass.reflection), h);
        h = fnv1a(&pass.defaultSampler, sizeof(pass.defaultSampler), h);
        auto bindCount = static_cast<std::uint32_t>(pass.bindMap.size());
        h = fnv1a(&bindCount, sizeof(bindCount), h);
        // XOR-combine bind entries (order-independent).
        std::uint64_t bindXor = 0;
        for (const auto& [name, entry] : pass.bindMap) {
            std::uint64_t entryH = fnv1a(name.data(), name.size());
            entryH = fnv1a(&entry.handle.index, sizeof(entry.handle.index), entryH);
            entryH = fnv1a(&entry.samplerOverride, sizeof(entry.samplerOverride), entryH);
            bindXor ^= entryH;
        }
        h = fnv1a(&bindXor, sizeof(bindXor), h);
    }

    // Hash each transient resource desc (dimensions determine alloc).
    for (const auto& res : resources) {
        if (res.tag != ResourceTag::Transient)
            continue;
        if (res.kind == ResourceKind::Image) {
            h = fnv1a(&res.imageDesc, sizeof(res.imageDesc), h);
        } else {
            h = fnv1a(&res.bufferDesc, sizeof(res.bufferDesc), h);
        }
    }

    return h;
}

Result<void> RenderGraph::compile() {
    using Clock = std::chrono::steady_clock;
    auto tStart = Clock::now();

    auto tResolve = tStart, tUsage = tStart, tAdj = tStart, tSort = tStart, tLifetime = tStart;
    const std::vector<std::uint32_t>* order = nullptr;

    // Resolve VK_REMAINING_* sentinels before hashing so identical
    // subresource ranges produce the same hash regardless of declaration style.
    resolveRemainingCounts();
    tResolve = Clock::now();
    accumulateTransientUsage();
    tUsage = Clock::now();

    auto graphHash = hashGraphStructure(passes_, resources_);
    bool cacheHit = (graphHash == lastGraphHash_ && !cachedOrder_.empty());

    if (cacheHit) {
        // Reuse cached topological order.
        order = &cachedOrder_;
        tAdj = tSort = tLifetime = Clock::now();
    } else {
        // Full compile path.
        buildAdjacency();
        tAdj = Clock::now();

        // Topological sort.
        auto sortResult = topologicalSort();
        if (!sortResult)
            return sortResult.error();
        cachedOrder_ = std::move(sortResult).value();
        order = &cachedOrder_;
        tSort = Clock::now();

        // Compute resource lifetimes.
        computeLifetimes(*order);
        tLifetime = Clock::now();

        // Cache for next frame.
        lastGraphHash_ = graphHash;
    }

    // Allocate transients (pool handles steady-state reuse).
    auto allocResult = allocateTransients();
    if (!allocResult)
        return allocResult.error();
    auto tAlloc = Clock::now();

    // Check handle stability: if all transient VkImage/VkBuffer handles are
    // identical to last frame, the compiled barriers are byte-identical.
    // Skip state reset and barrier compilation entirely.
    // External (imported) resources are excluded -- their handles change each
    // frame (e.g. round-robin swapchain images) but barrier structure is
    // determined by format and usage, not by which specific handle is imported.
    bool handlesStable = false;
    if (cacheHit && cachedImageHandles_.size() == resources_.size()) {
        handlesStable = true;
        for (std::uint32_t ri = 0; ri < static_cast<std::uint32_t>(resources_.size()); ++ri) {
            const auto& res = resources_[ri];
            if (res.tag != ResourceTag::Transient)
                continue;
            if (res.kind == ResourceKind::Image) {
                if (res.vkImage != cachedImageHandles_[ri]) {
                    handlesStable = false;
                    break;
                }
            } else {
                if (res.vkBuffer != cachedBufferHandles_[ri]) {
                    handlesStable = false;
                    break;
                }
            }
        }
    }

    auto us = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    auto tStateInit = tAlloc, tBarriers = tAlloc;

    if (handlesStable) {
        // Ultra-fast path: transient handles unchanged, barriers reusable.
        // Patch imported resource handles in cached barriers (e.g. swapchain
        // image changes each frame but barrier structure is identical).
        // Build old->new handle maps for external resources (typically 1-2).
        struct HandlePatch {
            VkImage oldImg;
            VkImage newImg;
            VkImageView oldView;
            VkImageView newView;
        };
        struct BufPatch {
            VkBuffer oldBuf;
            VkBuffer newBuf;
        };
        std::vector<HandlePatch> imgPatches;
        std::vector<BufPatch> bufPatches;
        for (std::uint32_t ri = 0; ri < static_cast<std::uint32_t>(resources_.size()); ++ri) {
            const auto& res = resources_[ri];
            if (res.tag != ResourceTag::External)
                continue;
            if (res.kind == ResourceKind::Image && res.vkImage != cachedImageHandles_[ri]) {
                imgPatches.push_back({cachedImageHandles_[ri], res.vkImage, cachedViewHandles_[ri],
                                      res.vkImageView});
                cachedImageHandles_[ri] = res.vkImage;
                cachedViewHandles_[ri] = res.vkImageView;
            } else if (res.kind == ResourceKind::Buffer &&
                       res.vkBuffer != cachedBufferHandles_[ri]) {
                bufPatches.push_back({cachedBufferHandles_[ri], res.vkBuffer});
                cachedBufferHandles_[ri] = res.vkBuffer;
            }
        }
        if (!imgPatches.empty() || !bufPatches.empty()) {
            for (auto& cp : compiledPasses_) {
                for (auto& ib : cp.barriers.imageBarriers) {
                    for (const auto& p : imgPatches) {
                        if (ib.image == p.oldImg) {
                            ib.image = p.newImg;
                            break;
                        }
                    }
                }
                for (auto& bb : cp.barriers.bufferBarriers) {
                    for (const auto& p : bufPatches) {
                        if (bb.buffer == p.oldBuf) {
                            bb.buffer = p.newBuf;
                            break;
                        }
                    }
                }
                // Patch Layer 1 resolved rendering views.
                for (auto& att : cp.rendering.colorAttachments) {
                    for (const auto& p : imgPatches) {
                        if (att.imageView == p.oldView) {
                            att.imageView = p.newView;
                            break;
                        }
                    }
                }
                if (cp.rendering.hasDepth) {
                    for (const auto& p : imgPatches) {
                        if (cp.rendering.depthAttachment.imageView == p.oldView) {
                            cp.rendering.depthAttachment.imageView = p.newView;
                            break;
                        }
                    }
                }
            }
        }

        // Patch clear values (may change frame-to-frame without structure change).
        for (auto& cp : compiledPasses_) {
            const auto& passDecl = passes_[cp.passIndex];
            for (const auto& ct : passDecl.colorTargets) {
                if (ct.index < cp.rendering.colorAttachments.size())
                    cp.rendering.colorAttachments[ct.index].clearValue.color = ct.clearValue;
            }
            if (passDecl.depthTarget && cp.rendering.hasDepth) {
                cp.rendering.depthAttachment.clearValue.depthStencil = {
                    passDecl.depthTarget->clearDepth,
                    passDecl.depthTarget->clearStencil,
                };
            }
        }

        // Descriptors are ephemeral -- must re-resolve every frame.
        auto descResult = resolveDescriptors();
        if (!descResult)
            return descResult;

        stats_ = cachedStats_;
        stats_.resolveUs = us(tStart, tResolve);
        stats_.usageUs = us(tResolve, tUsage);
        stats_.adjacencyUs = 0.0;
        stats_.sortUs = 0.0;
        stats_.lifetimeUs = 0.0;
        stats_.allocUs = us(tLifetime, tAlloc);
        stats_.stateInitUs = 0.0;
        stats_.barriersUs = 0.0;
        stats_.renderTargetUs = 0.0;
        stats_.descriptorUs = 0.0;
        stats_.statsUs = 0.0;
        auto tEnd = Clock::now();
        stats_.compileTimeUs = us(tStart, tEnd);
        isCompiled_ = true;
        return {};
    }

    // Initialize (or reset) state trackers.
    if (cacheHit && imageMaps_.size() == resources_.size()) {
        // Fast path: reset preserved maps in-place (no heap allocation).
        for (std::uint32_t ri = 0; ri < static_cast<std::uint32_t>(resources_.size()); ++ri) {
            const auto& res = resources_[ri];
            if (res.kind == ResourceKind::Image) {
                imageMaps_[ri].resetState(res.imageDesc.mipLevels, res.imageDesc.arrayLayers,
                                          res.initialState);
            } else {
                bufferStates_[ri] = res.initialState;
            }
        }
    } else {
        initStateTrackers();
    }
    tStateInit = Clock::now();

    // Compile barriers.
    if (cacheHit && compiledPasses_.size() == order->size()) {
        // Fast path: clear barrier batches (keeps vector capacity), recompile.
        for (auto& cp : compiledPasses_)
            cp.barriers.clear();
        auto barrierResult = compileBarriers(*order);
        if (!barrierResult)
            return barrierResult.error();
    } else {
        compiledPasses_.clear();
        auto barrierResult = compileBarriers(*order);
        if (!barrierResult)
            return barrierResult.error();
    }
    tBarriers = Clock::now();

    // Resolve render targets.
    resolveRenderTargets(*order);
    auto tRenderTargets = Clock::now();

    // Resolve descriptors.
    auto descResult = resolveDescriptors();
    if (!descResult)
        return descResult;
    auto tDescriptors = Clock::now();

    // Populate stats (lightweight: counts only, no barrier copies).
    stats_ = {};
    stats_.passCount = static_cast<std::uint32_t>(compiledPasses_.size());

    stats_.resolveUs = us(tStart, tResolve);
    stats_.usageUs = us(tResolve, tUsage);
    stats_.adjacencyUs = us(tUsage, tAdj);
    stats_.sortUs = us(tAdj, tSort);
    stats_.lifetimeUs = us(tSort, tLifetime);
    stats_.allocUs = us(tLifetime, tAlloc);
    stats_.stateInitUs = us(tAlloc, tStateInit);
    stats_.barriersUs = us(tStateInit, tBarriers);
    stats_.renderTargetUs = us(tBarriers, tRenderTargets);
    stats_.descriptorUs = us(tRenderTargets, tDescriptors);

    for (const auto& cp : compiledPasses_) {
        stats_.imageBarrierCount += static_cast<std::uint32_t>(cp.barriers.imageBarriers.size());
        stats_.bufferBarrierCount += static_cast<std::uint32_t>(cp.barriers.bufferBarriers.size());
    }

    stats_.transientCount = 0;
    for (const auto& res : resources_)
        if (res.tag == ResourceTag::Transient)
            stats_.transientCount++;

    auto tStatsEnd = Clock::now();
    stats_.statsUs = us(tDescriptors, tStatsEnd);
    stats_.compileTimeUs = us(tStart, tStatsEnd);

    // Cache handles and stats for next frame's stability check.
    cachedImageHandles_.resize(resources_.size());
    cachedViewHandles_.resize(resources_.size());
    cachedBufferHandles_.resize(resources_.size());
    for (std::uint32_t ri = 0; ri < static_cast<std::uint32_t>(resources_.size()); ++ri) {
        const auto& res = resources_[ri];
        cachedImageHandles_[ri] = res.vkImage;
        cachedViewHandles_[ri] = res.vkImageView;
        cachedBufferHandles_[ri] = res.vkBuffer;
    }
    cachedStats_ = stats_;

    isCompiled_ = true;
    return {};
}

void RenderGraph::execute(VkCommandBuffer cmd) {
    assert(isCompiled_ && "must call compile() before execute()");

    for (auto& cp : compiledPasses_) {
        // Emit barriers.
        if (!cp.barriers.empty()) {
            auto dep = cp.barriers.dependencyInfo();
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Collect image layouts for descriptor helpers.
        std::vector<PassResourceLayout> layouts;
        const auto& passDecl = passes_[cp.passIndex];
        for (const auto& acc : passDecl.accesses) {
            if (!acc.handle.valid())
                continue;
            if (resources_[acc.handle.index].kind == ResourceKind::Image) {
                layouts.push_back({acc.handle, acc.desiredState.currentLayout});
            }
        }

        // Invoke the pass callback.
        const ResolvedRendering* rendering = nullptr;
        if (!cp.rendering.colorAttachments.empty() || cp.rendering.hasDepth)
            rendering = &cp.rendering;

        const ResolvedDescriptors* descriptors = nullptr;
        if (passDecl.reflection != nullptr)
            descriptors = &cp.descriptors;

        PassContext ctx(resources_, std::move(layouts), rendering, descriptors);
        passDecl.recordFn(ctx, cmd);

#ifndef NDEBUG
        if (ctx.renderingActive())
            std::fprintf(stderr, "[vksdl::graph] pass '%s' did not call endRendering()\n",
                         passDecl.name.c_str());
#endif

        // Apply any state overrides from the callback.
        for (const auto& ov : ctx.overrides()) {
            if (!ov.handle.valid())
                continue;
            std::uint32_t ri = ov.handle.index;
            const auto& res = resources_[ri];

            if (res.kind == ResourceKind::Image) {
                if (ov.fullResource) {
                    // Reset the entire map to the override state.
                    imageMaps_[ri] = ImageSubresourceMap(res.imageDesc.mipLevels,
                                                         res.imageDesc.arrayLayers, ov.state);
                } else {
                    imageMaps_[ri].setState(ov.range, ov.state);
                }
            } else {
                bufferStates_[ri] = ov.state;
            }
        }
    }
}

Result<void> RenderGraph::compileAndExecute(VkCommandBuffer cmd) {
    auto result = compile();
    if (!result)
        return result;
    execute(cmd);
    return {};
}

Result<void> RenderGraph::prewarm() {
    auto result = compile();
    if (!result)
        return result;
    reset();
    return {};
}

void RenderGraph::reset() {
    recycleTransients();
    passes_.clear();
    resources_.clear();
    // imageMaps_, bufferStates_, compiledPasses_ preserved for cache-hit reuse.
    // adj_, inDegree_ preserved (only used on cache miss).
    // lastGraphHash_, cachedOrder_ preserved for structure cache.

    // Layer 2: reset descriptor allocator and destroy cached DSLs.
    if (descAllocator_)
        descAllocator_->resetPools();
    for (auto dsl : dslCache_)
        if (dsl != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, dsl, nullptr);
    dslCache_.clear();

    stats_ = {};
    isCompiled_ = false;
}

static const char* passTypeName(PassType t) {
    switch (t) {
    case PassType::Graphics:
        return "Graphics";
    case PassType::Compute:
        return "Compute";
    case PassType::Transfer:
        return "Transfer";
    }
    return "Unknown";
}

static const char* layoutName(VkImageLayout layout) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        return "UNDEFINED";
    case VK_IMAGE_LAYOUT_GENERAL:
        return "GENERAL";
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return "COLOR_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return "SHADER_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return "TRANSFER_SRC_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return "TRANSFER_DST_OPTIMAL";
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return "PRESENT_SRC_KHR";
    default:
        return "(other)";
    }
}

static void appendStageBits(std::string& out, VkPipelineStageFlags2 flags) {
    if (flags == VK_PIPELINE_STAGE_2_NONE) {
        out += "(none)";
        return;
    }

    bool first = true;
    auto add = [&](VkPipelineStageFlags2 bit, const char* name) {
        if (flags & bit) {
            if (!first)
                out += "|";
            out += name;
            first = false;
        }
    };

    add(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, "TOP_OF_PIPE");
    add(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, "DRAW_INDIRECT");
    add(VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, "VERTEX_INPUT");
    add(VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, "VERTEX_SHADER");
    add(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, "FRAGMENT_SHADER");
    add(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, "EARLY_FRAG");
    add(VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, "LATE_FRAG");
    add(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, "COLOR_ATTACHMENT_OUTPUT");
    add(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, "COMPUTE_SHADER");
    add(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, "ALL_TRANSFER");
    add(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, "BOTTOM_OF_PIPE");
    add(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, "ALL_GRAPHICS");
    add(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, "ALL_COMMANDS");
    add(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, "RT_SHADER");
    add(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, "AS_BUILD");

    if (first) {
        // Unrecognized bits.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%" PRIx64, static_cast<uint64_t>(flags));
        out += buf;
    }
}

static void appendAccessBits(std::string& out, VkAccessFlags2 flags) {
    if (flags == VK_ACCESS_2_NONE) {
        out += "(none)";
        return;
    }

    bool first = true;
    auto add = [&](VkAccessFlags2 bit, const char* name) {
        if (flags & bit) {
            if (!first)
                out += "|";
            out += name;
            first = false;
        }
    };

    add(VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, "INDIRECT_READ");
    add(VK_ACCESS_2_INDEX_READ_BIT, "INDEX_READ");
    add(VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT, "VERTEX_READ");
    add(VK_ACCESS_2_UNIFORM_READ_BIT, "UNIFORM_READ");
    add(VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT, "INPUT_ATTACHMENT_READ");
    add(VK_ACCESS_2_SHADER_READ_BIT, "SHADER_READ");
    add(VK_ACCESS_2_SHADER_WRITE_BIT, "SHADER_WRITE");
    add(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT, "COLOR_ATTACHMENT_READ");
    add(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, "COLOR_ATTACHMENT_WRITE");
    add(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, "DEPTH_STENCIL_READ");
    add(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, "DEPTH_STENCIL_WRITE");
    add(VK_ACCESS_2_TRANSFER_READ_BIT, "TRANSFER_READ");
    add(VK_ACCESS_2_TRANSFER_WRITE_BIT, "TRANSFER_WRITE");
    add(VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, "SHADER_SAMPLED_READ");
    add(VK_ACCESS_2_SHADER_STORAGE_READ_BIT, "SHADER_STORAGE_READ");
    add(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, "SHADER_STORAGE_WRITE");

    if (first) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "0x%" PRIx64, static_cast<uint64_t>(flags));
        out += buf;
    }
}

static const char* formatName(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "B8G8R8A8_SRGB";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "R16G16B16A16_SFLOAT";
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return "R32G32B32A32_SFLOAT";
    case VK_FORMAT_D32_SFLOAT:
        return "D32_SFLOAT";
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return "D24_UNORM_S8_UINT";
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return "D32_SFLOAT_S8_UINT";
    case VK_FORMAT_D16_UNORM:
        return "D16_UNORM";
    default:
        return "(other format)";
    }
}

// Look up resource name by VkImage handle.
static const char* findResourceName(const std::vector<ResourceEntry>& resources, VkImage image) {
    for (const auto& r : resources) {
        if (r.kind == ResourceKind::Image && r.vkImage == image && !r.name.empty())
            return r.name.c_str();
    }
    return nullptr;
}

static const char* findBufferName(const std::vector<ResourceEntry>& resources, VkBuffer buffer) {
    for (const auto& r : resources) {
        if (r.kind == ResourceKind::Buffer && r.vkBuffer == buffer && !r.name.empty())
            return r.name.c_str();
    }
    return nullptr;
}

void RenderGraph::dumpLog() const {
    if (!isCompiled_) {
        std::fprintf(stderr, "[vksdl::graph] Not compiled yet.\n");
        return;
    }

    std::fprintf(stderr, "[vksdl::graph] Compiled %u passes:\n", stats_.passCount);
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(compiledPasses_.size()); ++i) {
        const auto& decl = passes_[compiledPasses_[i].passIndex];
        std::fprintf(stderr, "  [%u] %-20s (%s)\n", i, decl.name.c_str(), passTypeName(decl.type));
    }

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(compiledPasses_.size()); ++i) {
        const auto& cp = compiledPasses_[i];
        const auto& barriers = cp.barriers;
        if (barriers.empty())
            continue;

        std::fprintf(stderr, "[vksdl::graph] pass '%s':\n", passes_[cp.passIndex].name.c_str());

        for (const auto& b : barriers.imageBarriers) {
            const char* rname = findResourceName(resources_, b.image);
            std::string nameStr = rname ? rname : "(unnamed)";

            std::string srcStage, srcAccess, dstStage, dstAccess;
            appendStageBits(srcStage, b.srcStageMask);
            appendStageBits(dstStage, b.dstStageMask);
            appendAccessBits(dstAccess, b.dstAccessMask);

            // Show "(already visible)" when srcAccess==0 but srcStage!=0.
            // This is the multi-reader fan-out signal: the write was already
            // made visible by a prior reader's barrier, only execution dep needed.
            if (b.srcAccessMask == VK_ACCESS_2_NONE && b.srcStageMask != VK_PIPELINE_STAGE_2_NONE)
                srcAccess = "(already visible)";
            else
                appendAccessBits(srcAccess, b.srcAccessMask);

            std::fprintf(stderr,
                         "  IMG barrier: %-18s %s -> %s\n"
                         "               src: %s / %s\n"
                         "               dst: %s / %s\n",
                         nameStr.c_str(), layoutName(b.oldLayout), layoutName(b.newLayout),
                         srcStage.c_str(), srcAccess.c_str(), dstStage.c_str(), dstAccess.c_str());
        }

        for (const auto& b : barriers.bufferBarriers) {
            const char* rname = findBufferName(resources_, b.buffer);
            std::string nameStr = rname ? rname : "(unnamed)";

            std::string srcStage, srcAccess, dstStage, dstAccess;
            appendStageBits(srcStage, b.srcStageMask);
            appendStageBits(dstStage, b.dstStageMask);
            appendAccessBits(dstAccess, b.dstAccessMask);

            if (b.srcAccessMask == VK_ACCESS_2_NONE && b.srcStageMask != VK_PIPELINE_STAGE_2_NONE)
                srcAccess = "(already visible)";
            else
                appendAccessBits(srcAccess, b.srcAccessMask);

            std::fprintf(stderr,
                         "  BUF barrier: %-18s\n"
                         "               src: %s / %s\n"
                         "               dst: %s / %s\n",
                         nameStr.c_str(), srcStage.c_str(), srcAccess.c_str(), dstStage.c_str(),
                         dstAccess.c_str());
        }
    }

    if (stats_.transientCount > 0) {
        std::fprintf(stderr, "[vksdl::graph] Transient allocations:\n");
        for (const auto& res : resources_) {
            if (res.tag != ResourceTag::Transient)
                continue;
            const char* name = res.name.empty() ? "(unnamed)" : res.name.c_str();
            if (res.kind == ResourceKind::Image) {
                std::uint64_t handle{};
                std::memcpy(&handle, &res.vkImage, sizeof(handle));
                std::fprintf(stderr, "  %-18s %ux%u  %-24s (VkImage 0x%" PRIx64 ")\n", name,
                             res.imageDesc.width, res.imageDesc.height,
                             formatName(res.imageDesc.format), handle);
            } else {
                std::uint64_t handle{};
                std::memcpy(&handle, &res.vkBuffer, sizeof(handle));
                std::fprintf(stderr, "  %-18s %llu bytes (VkBuffer 0x%" PRIx64 ")\n", name,
                             static_cast<unsigned long long>(res.bufferDesc.size), handle);
            }
        }
    }

    std::fprintf(stderr,
                 "[vksdl::graph] compile %.0fus: resolve %.0fus, usage %.0fus, "
                 "adjacency %.0fus, sort %.0fus,\n"
                 "               lifetime %.0fus, alloc %.0fus, stateInit %.0fus, "
                 "barriers %.0fus, renderTargets %.0fus, descriptors %.0fus, stats %.0fus\n",
                 stats_.compileTimeUs, stats_.resolveUs, stats_.usageUs, stats_.adjacencyUs,
                 stats_.sortUs, stats_.lifetimeUs, stats_.allocUs, stats_.stateInitUs,
                 stats_.barriersUs, stats_.renderTargetUs, stats_.descriptorUs, stats_.statsUs);

    std::fprintf(stderr,
                 "[vksdl::graph] %u barriers (%u image, %u buffer), "
                 "%u passes, %u transients\n",
                 stats_.imageBarrierCount + stats_.bufferBarrierCount, stats_.imageBarrierCount,
                 stats_.bufferBarrierCount, stats_.passCount, stats_.transientCount);
}

} // namespace vksdl::graph
