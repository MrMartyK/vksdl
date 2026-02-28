#pragma once

#include <vksdl/error.hpp>
#include <vksdl/pipeline_model/pipeline_handle.hpp>
#include <vksdl/pipeline_model/pipeline_policy.hpp>
#include <vksdl/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vksdl {

class Device;
class Pipeline;
class PipelineBuilder;
class PipelineCache;

// Central async pipeline compilation engine.
//
// Implements a three-step pipeline acquisition that eliminates shader stutter:
// 1. Cache probe -- FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT (zero cost if cached)
// 2. Fast-link -- Assemble from pre-compiled GPL library parts (GPL path only)
// 3. Background optimize -- Submit fully optimized monolithic pipeline on
//    worker thread, atomically swap in when ready
//
// For the monolithic path (no GPL): step 1 is the cache probe, step 2 is
// skipped, step 3 compiles a monolithic pipeline synchronously (first time)
// or returns the cached pipeline (subsequent times).
//
// Shader modules, descriptor set layouts, and pipeline layouts passed to the
// PipelineBuilder must remain valid until waitIdle() returns or the
// PipelineCompiler is destroyed.
//
// ~PipelineCompiler blocks until all background compilations complete.
// Call waitIdle() first if you need predictable shutdown timing.
//
// Thread safety: thread-confined. compile() and waitIdle() must be called
// from a single thread. Background optimization is internal.
class PipelineCompiler {
  public:
    [[nodiscard]] static Result<PipelineCompiler>
    create(const Device& device, PipelineCache& cache,
           PipelinePolicy policy = PipelinePolicy::Auto);

    ~PipelineCompiler();
    PipelineCompiler(PipelineCompiler&&) noexcept;
    PipelineCompiler& operator=(PipelineCompiler&&) noexcept;
    PipelineCompiler(const PipelineCompiler&) = delete;
    PipelineCompiler& operator=(const PipelineCompiler&) = delete;

    // Blocks until a usable pipeline exists. Only optimization is async.
    // The builder is not modified -- the compiler reads its state.
    [[nodiscard]] Result<PipelineHandle> compile(const PipelineBuilder& builder);

    void waitIdle();

    [[nodiscard]] std::uint32_t pendingCount() const;

    // The resolved pipeline model (stable after create()).
    [[nodiscard]] PipelineModel resolvedModel() const;

    [[nodiscard]] PipelinePolicy policy() const;

    [[nodiscard]] PipelineModelInfo modelInfo() const;

  private:
    PipelineCompiler() = default;
    void destroy();

    // Transfer Pipeline ownership into a PipelineHandleImpl.
    // Must be a member (not a free function) so friend access to Pipeline works.
    static void* transferPipeline(VkDevice device, Pipeline& pipeline, bool markOptimized);

    // Opaque impl hides threading primitives from public header.
    void* impl_ = nullptr;
};

} // namespace vksdl
