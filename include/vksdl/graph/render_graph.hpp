#pragma once

#include <vksdl/graph/barrier_compiler.hpp>
#include <vksdl/graph/pass.hpp>
#include <vksdl/graph/pass_context.hpp>
#include <vksdl/graph/resource.hpp>
#include <vksdl/graph/resource_state.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vksdl {
class Device;
class Allocator;
class Image;
class Buffer;
struct ReflectedLayout;
class DescriptorAllocator;
} // namespace vksdl

namespace vksdl::graph {

// Pre-resolved rendering state for one pass (Layer 1).
// Assembled during compile() from ColorTargetDecl / DepthTargetDecl.
// Consumed by PassContext::beginRendering() during execute().
struct ResolvedRendering {
    std::vector<VkRenderingAttachmentInfo> colorAttachments;
    VkRenderingAttachmentInfo depthAttachment{};
    bool hasDepth = false;
    VkExtent2D renderArea{};
};

// Pre-resolved descriptor state for one pass (Layer 2).
// Assembled during compile() from ReflectedLayout + bind map.
// Consumed by PassContext::bindPipeline() / bindDescriptors() during execute().
struct ResolvedDescriptors {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    std::vector<VkDescriptorSet> sets; // per set index; VK_NULL_HANDLE = unmanaged
};

// Compiled pass: sorted index + pre-computed barriers + optional rendering/descriptor state.
struct CompiledPass {
    std::uint32_t passIndex;         // index into passes_
    BarrierBatch barriers;           // barriers to emit before this pass
    ResolvedRendering rendering;     // Layer 1: pre-resolved VkRenderingInfo (empty if Layer 0)
    ResolvedDescriptors descriptors; // Layer 2: auto-resolved descriptors (empty if Layer 0/1)
};

// Transient VMA-backed image created during compile().
// Stores desc for pool matching across frames.
struct TransientImage {
    ImageDesc desc;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    void* allocation = nullptr; // VmaAllocation stored as void*
};

// Transient VMA-backed buffer created during compile().
struct TransientBuffer {
    BufferDesc desc;
    VkBuffer buffer = VK_NULL_HANDLE;
    void* allocation = nullptr; // VmaAllocation stored as void*
};

// Aggregate compile/execute statistics.
struct GraphStats {
    std::uint32_t passCount = 0;
    std::uint32_t imageBarrierCount = 0;
    std::uint32_t bufferBarrierCount = 0;
    std::uint32_t transientCount = 0;
    double compileTimeUs = 0.0;

    // Timing breakdown (microseconds).
    double resolveUs = 0.0;
    double usageUs = 0.0;
    double adjacencyUs = 0.0;
    double sortUs = 0.0;
    double lifetimeUs = 0.0;
    double allocUs = 0.0;
    double stateInitUs = 0.0;
    double barriersUs = 0.0;
    double renderTargetUs = 0.0;
    double descriptorUs = 0.0;
    double statsUs = 0.0;
};

// Render graph: declare passes with resource dependencies, compile to
// automatic barrier insertion, execute.
//
// Usage:
//   graph.importImage(img, state);
//   auto tex = graph.createImage(desc);
//   graph.addPass("geometry", PassType::Graphics, setupFn, recordFn);
//   graph.compile();
//   graph.execute(cmd);
//   graph.reset(); // call each frame before re-declaring
//
// Thread safety: thread-confined. All methods (addPass/compile/execute/reset)
// must be called from the same thread.
class RenderGraph {
  public:
    RenderGraph(const Device& device, const Allocator& allocator);
    ~RenderGraph();

    RenderGraph(RenderGraph&&) noexcept;
    RenderGraph& operator=(RenderGraph&&) noexcept;
    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;

    // Import an external image. Aspect auto-derived from format.
    [[nodiscard]] ResourceHandle importImage(VkImage image, VkImageView view, VkFormat format,
                                             std::uint32_t width, std::uint32_t height,
                                             const ResourceState& initialState,
                                             std::uint32_t mipLevels = 1,
                                             std::uint32_t arrayLayers = 1,
                                             std::string_view name = "");

    // Import from a vksdl::Image.
    [[nodiscard]] ResourceHandle importImage(const Image& image, const ResourceState& initialState,
                                             std::string_view name = "");

    // Import an external buffer.
    [[nodiscard]] ResourceHandle importBuffer(VkBuffer buffer, VkDeviceSize size,
                                              const ResourceState& initialState,
                                              std::string_view name = "");

    // Import from a vksdl::Buffer.
    [[nodiscard]] ResourceHandle importBuffer(const Buffer& buffer,
                                              const ResourceState& initialState,
                                              std::string_view name = "");

    // Declare a transient image (allocated at compile time).
    [[nodiscard]] ResourceHandle createImage(const ImageDesc& desc, std::string_view name = "");

    // Declare a transient buffer (allocated at compile time).
    [[nodiscard]] ResourceHandle createBuffer(const BufferDesc& desc, std::string_view name = "");

    // Layer 0/1: declare a pass with explicit resource access declarations.
    void addPass(std::string_view name, PassType type, SetupFn setup, RecordFn record);

    // Layer 2: pipeline-aware pass with auto-bind. SPIR-V reflection infers
    // resource accesses from bind() entries; descriptors auto-allocated/written.
    // `reflection` must outlive the graph frame (typically init-time static).
    void addPass(std::string_view name, PassType type, VkPipeline pipeline,
                 VkPipelineLayout pipelineLayout, const ReflectedLayout& reflection, SetupFn setup,
                 RecordFn record);

    // Compile the graph: topo sort, allocate transients, compute barriers.
    [[nodiscard]] Result<void> compile();

    // Execute all compiled passes, emitting barriers and invoking callbacks.
    void execute(VkCommandBuffer cmd);

    // Convenience: compile + execute.
    [[nodiscard]] Result<void> compileAndExecute(VkCommandBuffer cmd);

    // Pre-warm the transient pool by compiling the current graph and
    // recycling allocations. Call after declaring the graph structure
    // (importImage/createImage/addPass) during init, before the main loop.
    // After prewarm(), call reset() to clear the graph for the first real frame.
    // This eliminates VMA allocation cost from the first compile in the loop.
    [[nodiscard]] Result<void> prewarm();

    // Destroy transients and clear all state. Call once per frame before
    // re-importing resources and declaring passes.
    void reset();

    [[nodiscard]] std::uint32_t passCount() const {
        return static_cast<std::uint32_t>(passes_.size());
    }
    [[nodiscard]] std::uint32_t resourceCount() const {
        return static_cast<std::uint32_t>(resources_.size());
    }
    [[nodiscard]] bool isCompiled() const {
        return isCompiled_;
    }

    // Graph statistics populated by compile(). Valid until reset().
    [[nodiscard]] const GraphStats& stats() const {
        return stats_;
    }

    // Print a detailed debug log to stderr. Call after compile().
    // Shows pass execution order, barriers per pass, transient allocations,
    // and a one-line summary. Use for debugging synchronization issues.
    void dumpLog() const;

  private:
    void destroy();
    void destroyTransients();
    void recycleTransients(); // move active transients to pool (no VMA calls)
    void destroyPool();       // destroy pooled allocations via VMA

    // Compile steps.
    void resolveRemainingCounts();
    void accumulateTransientUsage();
    void buildAdjacency();
    [[nodiscard]] Result<std::vector<std::uint32_t>> topologicalSort();
    void computeLifetimes(const std::vector<std::uint32_t>& order);
    [[nodiscard]] Result<void> allocateTransients();
    void initStateTrackers();
    [[nodiscard]] Result<void> compileBarriers(const std::vector<std::uint32_t>& order);
    void resolveRenderTargets(const std::vector<std::uint32_t>& order);
    [[nodiscard]] Result<void> resolveDescriptors();

    VkDevice device_ = VK_NULL_HANDLE;
    void* allocator_ = nullptr; // VmaAllocator, stored as void*
    bool hasUnifiedLayouts_ = false;

    std::vector<PassDecl> passes_;
    std::vector<ResourceEntry> resources_;
    std::vector<ImageSubresourceMap> imageMaps_; // parallel to resources_ (image entries only)
    std::vector<ResourceState> bufferStates_;    // parallel to resources_ (buffer entries only)

    // Adjacency (populated by buildAdjacency, consumed by topologicalSort).
    std::vector<std::vector<std::uint32_t>> adj_;
    std::vector<std::uint32_t> inDegree_;

    // Compiled output.
    std::vector<CompiledPass> compiledPasses_;
    bool isCompiled_ = false;
    GraphStats stats_;

    // Transient allocations (destroyed in destructor, recycled on reset).
    std::vector<TransientImage> transientImages_;
    std::vector<TransientBuffer> transientBuffers_;

    // Transient pool (reused across frames when desc matches).
    std::vector<TransientImage> imagePool_;
    std::vector<TransientBuffer> bufferPool_;

    // Graph structure cache: skip re-sorting on identical frames.
    std::uint64_t lastGraphHash_ = 0;
    std::vector<std::uint32_t> cachedOrder_; // topological sort from previous compile

    // Handle stability cache: when all transient VkImage/VkBuffer handles
    // match last frame, compiled barriers are byte-identical -- skip P7+P8.
    std::vector<VkImage> cachedImageHandles_;
    std::vector<VkImageView> cachedViewHandles_; // parallel, for Layer 1 patching
    std::vector<VkBuffer> cachedBufferHandles_;
    GraphStats cachedStats_;

    // Layer 2: descriptor auto-bind state.
    std::unique_ptr<DescriptorAllocator> descAllocator_;
    std::vector<VkDescriptorSetLayout> dslCache_; // created per compile, destroyed on reset/destroy
};

} // namespace vksdl::graph
