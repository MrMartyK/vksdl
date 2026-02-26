#pragma once

#include <vulkan/vulkan.h>

namespace vksdl {

// Atomic-swap pipeline wrapper for background optimization.
//
// After PipelineCompiler::compile() returns, the handle holds a usable pipeline
// (either cache-hit monolithic or fast-linked GPL). A background thread may be
// compiling a fully optimized monolithic pipeline. When that completes, bind()
// automatically uses the optimized version via atomic swap.
//
// Both baseline and optimized pipelines are kept alive until ~PipelineHandle().
// This wastes a small amount of driver memory (~kilobytes of metadata) but
// eliminates frame-indexed retirement complexity.
//
// Thread safety: bind() and isOptimized() are safe to call from any thread
// (atomic acquire on the optimized pipeline pointer). Destruction is not
// concurrent with bind().
class PipelineHandle {
public:
    ~PipelineHandle();
    PipelineHandle(PipelineHandle&&) noexcept;
    PipelineHandle& operator=(PipelineHandle&&) noexcept;
    PipelineHandle(const PipelineHandle&) = delete;
    PipelineHandle& operator=(const PipelineHandle&) = delete;

    // Bind the best available pipeline to a command buffer.
    // Uses atomic acquire to check for background-compiled optimized pipeline.
    void bind(VkCommandBuffer cmd) const;

    // True when the background-compiled optimized pipeline has been swapped in.
    [[nodiscard]] bool isOptimized() const;

    // True immediately after compile() returns (always true for valid handles).
    [[nodiscard]] bool isReady() const;

    // Best available pipeline handle (optimized if ready, baseline otherwise).
    [[nodiscard]] VkPipeline       vkPipeline()      const;
    [[nodiscard]] VkPipelineLayout vkPipelineLayout() const;

private:
    friend class PipelineCompiler;
    PipelineHandle() = default;

    void destroy();

    // Opaque impl hidden from public header (avoids <atomic> include).
    void* impl_ = nullptr;
};

} // namespace vksdl
