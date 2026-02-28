#include <vksdl/device.hpp>
#include <vksdl/pipeline.hpp>
#include <vksdl/pipeline_cache.hpp>
#include <vksdl/pipeline_model/gpl_library.hpp>
#include <vksdl/pipeline_model/pipeline_compiler.hpp>

#include "pipeline_handle_impl.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vksdl {

namespace detail {

struct CompileTask {
    std::function<void()> work;
};

// FNV-1a hash for pipeline identity. Not cryptographic, just collision-avoidance.
class HashBuilder {
  public:
    void feed(const void* data, std::size_t size) {
        auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash_ ^= bytes[i];
            hash_ *= 0x100000001b3ULL;
        }
    }

    template <typename T> void feed(const T& value) {
        feed(&value, sizeof(T));
    }

    [[nodiscard]] std::uint64_t finish() const {
        return hash_;
    }

  private:
    std::uint64_t hash_ = 0xcbf29ce484222325ULL; // FNV offset basis
};

struct PipelineCompilerImpl {
    VkDevice device = VK_NULL_HANDLE;
    const Device* devicePtr = nullptr; // non-owning, valid for lifetime of compiler
    VkPipelineCache cache = VK_NULL_HANDLE;
    PipelinePolicy policy = PipelinePolicy::Auto;
    PipelineModel resolvedModel = PipelineModel::Monolithic;
    PipelineModelInfo info;

    // Worker thread pool.
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<CompileTask> tasks;
    std::atomic<bool> running{true};
    std::atomic<std::uint32_t> pending{0};

    // GPL library caches (keyed by content hash).
    std::shared_mutex viCacheMutex, prCacheMutex, fsCacheMutex, foCacheMutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<GplLibrary>> vertexInputCache;
    std::unordered_map<std::uint64_t, std::shared_ptr<GplLibrary>> preRasterCache;
    std::unordered_map<std::uint64_t, std::shared_ptr<GplLibrary>> fragmentShaderCache;
    std::unordered_map<std::uint64_t, std::shared_ptr<GplLibrary>> fragmentOutputCache;

    void workerLoop() {
        while (true) {
            CompileTask task;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [this] {
                    return !running.load(std::memory_order_relaxed) || !tasks.empty();
                });
                if (!running.load(std::memory_order_relaxed) && tasks.empty())
                    return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task.work();
            pending.fetch_sub(1, std::memory_order_release);
        }
    }

    void enqueue(CompileTask task) {
        pending.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex);
            tasks.push(std::move(task));
        }
        cv.notify_one();
    }

    void shutdown() {
        running.store(false, std::memory_order_release);
        cv.notify_all();
        for (auto& w : workers) {
            if (w.joinable())
                w.join();
        }
        workers.clear();
    }
};

} // namespace detail

PipelineCompiler::~PipelineCompiler() {
    destroy();
}

PipelineCompiler::PipelineCompiler(PipelineCompiler&& o) noexcept : impl_(o.impl_) {
    o.impl_ = nullptr;
}

PipelineCompiler& PipelineCompiler::operator=(PipelineCompiler&& o) noexcept {
    if (this != &o) {
        destroy();
        impl_ = o.impl_;
        o.impl_ = nullptr;
    }
    return *this;
}

void PipelineCompiler::destroy() {
    if (!impl_)
        return;
    auto* impl = static_cast<detail::PipelineCompilerImpl*>(impl_);
    impl->shutdown();
    delete impl;
    impl_ = nullptr;
}

Result<PipelineCompiler> PipelineCompiler::create(const Device& device, PipelineCache& cache,
                                                  PipelinePolicy policy) {

    if (policy == PipelinePolicy::ForceShaderObject) {
        return Error{"create pipeline compiler", 0, "shader object support not yet implemented"};
    }

    auto* impl = new detail::PipelineCompilerImpl;
    impl->device = device.vkDevice();
    impl->devicePtr = &device;
    impl->cache = cache.vkPipelineCache();
    impl->policy = policy;

    impl->info.hasPCCC = device.hasPipelineCreationCacheControl();
    impl->info.hasGPL = device.hasGPL();
    impl->info.fastLink = device.hasGplFastLinking();

    switch (policy) {
    case PipelinePolicy::Auto:
        if (device.hasGPL() && device.hasGplFastLinking() &&
            device.hasGplIndependentInterpolation()) {
            impl->resolvedModel = PipelineModel::GPL;
            impl->info.model = PipelineModel::GPL;
        } else {
            impl->resolvedModel = PipelineModel::Monolithic;
            impl->info.model = PipelineModel::Monolithic;
#ifndef NDEBUG
            if (device.hasGPL() && !device.hasGplIndependentInterpolation()) {
                std::fprintf(stderr, "[vksdl] GPL available but graphicsPipelineLibrary"
                                     "IndependentInterpolationDecoration is VK_FALSE. "
                                     "Falling back to monolithic pipelines.\n");
            }
#endif
        }
        break;

    case PipelinePolicy::ForceMonolithic:
        impl->resolvedModel = PipelineModel::Monolithic;
        impl->info.model = PipelineModel::Monolithic;
        break;

    case PipelinePolicy::PreferGPL:
        if (device.hasGPL()) {
            impl->resolvedModel = PipelineModel::GPL;
            impl->info.model = PipelineModel::GPL;
        } else {
            return Error{"create pipeline compiler", 0,
                         "PreferGPL policy requires VK_EXT_graphics_pipeline_library "
                         "but it is not available on this device"};
        }
        break;

    default:
        break;
    }

    std::uint32_t threadCount = 1;
    if (impl->resolvedModel == PipelineModel::GPL) {
        auto hw = std::thread::hardware_concurrency();
        threadCount = std::max(1u, hw / 2);
    }
    for (std::uint32_t i = 0; i < threadCount; ++i) {
        impl->workers.emplace_back(&detail::PipelineCompilerImpl::workerLoop, impl);
    }

    PipelineCompiler compiler;
    compiler.impl_ = impl;
    return compiler;
}

// Private static member of PipelineCompiler so friend access to Pipeline works.
void* PipelineCompiler::transferPipeline(VkDevice device, Pipeline& pipeline, bool markOptimized) {
    auto* hi = new detail::PipelineHandleImpl;
    hi->device = device;
    hi->baseline = pipeline.pipeline_;
    hi->optimized.store(VK_NULL_HANDLE, std::memory_order_relaxed);
    hi->layout = pipeline.layout_;
    hi->ownsLayout = pipeline.ownsLayout_;
    hi->bindPoint = pipeline.bindPoint_;

    if (markOptimized) {
        hi->optimized.store(hi->baseline, std::memory_order_release);
    }

    // Prevent Pipeline destructor from destroying transferred handles.
    pipeline.pipeline_ = VK_NULL_HANDLE;
    pipeline.layout_ = VK_NULL_HANDLE;

    return hi;
}

Result<PipelineHandle> PipelineCompiler::compile(const PipelineBuilder& builder) {
    auto* impl = static_cast<detail::PipelineCompilerImpl*>(impl_);

    // Monolithic path: synchronous compilation with optional cache probe.
    if (impl->resolvedModel == PipelineModel::Monolithic) {

        // Step 1: Cache probe (zero-cost if cached).
        if (impl->info.hasPCCC) {
            auto probeResult =
                builder.buildWithFlags(VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT);
            if (probeResult.ok()) {
                Pipeline pipeline = std::move(probeResult).value();
                auto* hi = transferPipeline(impl->device, pipeline, true);

                PipelineHandle handle;
                handle.impl_ = hi;
                return handle;
            }
            // VK_PIPELINE_COMPILE_REQUIRED is expected -- fall through.
            auto vr = static_cast<VkResult>(probeResult.error().vkResult);
            if (vr != VK_PIPELINE_COMPILE_REQUIRED) {
                return std::move(probeResult).error();
            }
        }

        // Step 2 (monolithic): Build synchronously.
        auto buildResult = builder.buildWithFlags(0);
        if (!buildResult.ok()) {
            return std::move(buildResult).error();
        }

        Pipeline pipeline = std::move(buildResult).value();
        auto* hi = transferPipeline(impl->device, pipeline, true);

        PipelineHandle handle;
        handle.impl_ = hi;
        return handle;
    }

    // GPL path: 3-step acquisition.

    // Step 1: Cache probe (if PCCC available).
    if (impl->info.hasPCCC) {
        auto probeResult =
            builder.buildWithFlags(VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT);
        if (probeResult.ok()) {
            Pipeline pipeline = std::move(probeResult).value();
            auto* hi = transferPipeline(impl->device, pipeline, true);

            PipelineHandle handle;
            handle.impl_ = hi;
            return handle;
        }
        auto vr = static_cast<VkResult>(probeResult.error().vkResult);
        if (vr != VK_PIPELINE_COMPILE_REQUIRED) {
            return std::move(probeResult).error();
        }
    }

    // Step 2: Build GPL library parts and fast-link.
    // All builder member access here is valid because PipelineCompiler
    // is a friend of PipelineBuilder.

    auto vertBindings = builder.vertexBindings_;
    auto vertAttributes = builder.vertexAttributes_;
    auto topology = builder.topology_;
    auto polygonMode = builder.polygonMode_;
    auto cullMode = builder.cullMode_;
    auto frontFace = builder.frontFace_;
    auto colorFormat = builder.colorFormat_;
    auto depthFormat = builder.depthFormat_;
    auto depthCompareOp = builder.depthCompareOp_;
    auto samples = builder.samples_;
    auto enableBlending = builder.enableBlending_;
    auto extraDynamicStates = builder.extraDynamicStates_;
    auto dsLayouts = builder.descriptorSetLayouts_;
    auto pcRanges = builder.pushConstantRanges_;
    auto externalLayout = builder.externalLayout_;

    std::vector<std::uint32_t> vertCode, fragCode;
    VkShaderModule vertMod = builder.vertModule_;
    VkShaderModule fragMod = builder.fragModule_;
    bool createdVert = false, createdFrag = false;

    auto destroyModules = [&]() {
        if (createdVert && vertMod != VK_NULL_HANDLE)
            vkDestroyShaderModule(impl->device, vertMod, nullptr);
        if (createdFrag && fragMod != VK_NULL_HANDLE)
            vkDestroyShaderModule(impl->device, fragMod, nullptr);
    };

    if (vertMod == VK_NULL_HANDLE) {
        auto code = readSpv(builder.vertPath_);
        if (!code.ok())
            return std::move(code).error();
        vertCode = std::move(code).value();

        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = vertCode.size() * sizeof(std::uint32_t);
        ci.pCode = vertCode.data();
        VkResult vr = vkCreateShaderModule(impl->device, &ci, nullptr, &vertMod);
        if (vr != VK_SUCCESS) {
            return Error{"create vertex shader module", static_cast<std::int32_t>(vr),
                         "vkCreateShaderModule failed"};
        }
        createdVert = true;
    }

    if (fragMod == VK_NULL_HANDLE) {
        auto code = readSpv(builder.fragPath_);
        if (!code.ok()) {
            destroyModules();
            return std::move(code).error();
        }
        fragCode = std::move(code).value();

        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = fragCode.size() * sizeof(std::uint32_t);
        ci.pCode = fragCode.data();
        VkResult vr = vkCreateShaderModule(impl->device, &ci, nullptr, &fragMod);
        if (vr != VK_SUCCESS) {
            destroyModules();
            return Error{"create fragment shader module", static_cast<std::int32_t>(vr),
                         "vkCreateShaderModule failed"};
        }
        createdFrag = true;
    }

    VkPipelineLayout pipelineLayout = externalLayout;
    bool ownsLayout = false;

    if (pipelineLayout == VK_NULL_HANDLE) {
        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount = static_cast<std::uint32_t>(dsLayouts.size());
        layoutCI.pSetLayouts = dsLayouts.empty() ? nullptr : dsLayouts.data();
        layoutCI.pushConstantRangeCount = static_cast<std::uint32_t>(pcRanges.size());
        layoutCI.pPushConstantRanges = pcRanges.empty() ? nullptr : pcRanges.data();

        VkResult vr = vkCreatePipelineLayout(impl->device, &layoutCI, nullptr, &pipelineLayout);
        if (vr != VK_SUCCESS) {
            destroyModules();
            return Error{"create pipeline layout", static_cast<std::int32_t>(vr),
                         "vkCreatePipelineLayout failed"};
        }
        ownsLayout = true;
    }

    auto computeHash = [](auto feedFn) -> std::uint64_t {
        detail::HashBuilder h;
        feedFn(h);
        return h.finish();
    };

    auto viHash = computeHash([&](detail::HashBuilder& h) {
        for (const auto& vb : vertBindings)
            h.feed(vb);
        for (const auto& va : vertAttributes)
            h.feed(va);
        h.feed(topology);
    });
    auto prHash = computeHash([&](detail::HashBuilder& h) {
        if (!vertCode.empty()) {
            h.feed(vertCode.data(), vertCode.size() * sizeof(std::uint32_t));
        } else {
            h.feed(vertMod);
        }
        h.feed(polygonMode);
        h.feed(cullMode);
        h.feed(frontFace);
    });
    auto fsHash = computeHash([&](detail::HashBuilder& h) {
        if (!fragCode.empty()) {
            h.feed(fragCode.data(), fragCode.size() * sizeof(std::uint32_t));
        } else {
            h.feed(fragMod);
        }
        h.feed(depthFormat);
        h.feed(depthCompareOp);
    });
    auto foHash = computeHash([&](detail::HashBuilder& h) {
        h.feed(colorFormat);
        h.feed(depthFormat);
        h.feed(samples);
        h.feed(enableBlending);
    });

    // Helper: lookup-or-create a cached library part.
    auto getOrCreate = [](std::shared_mutex& mtx,
                          std::unordered_map<std::uint64_t, std::shared_ptr<GplLibrary>>& theCache,
                          std::uint64_t hash, auto buildFn) -> Result<std::shared_ptr<GplLibrary>> {
        {
            std::shared_lock rlock(mtx);
            auto it = theCache.find(hash);
            if (it != theCache.end())
                return it->second;
        }
        std::unique_lock wlock(mtx);
        auto it = theCache.find(hash);
        if (it != theCache.end())
            return it->second;

        auto result = buildFn();
        if (!result.ok())
            return std::move(result).error();

        auto ptr = std::make_shared<GplLibrary>(std::move(result).value());
        theCache.emplace(hash, ptr);
        return ptr;
    };

    auto viResult = getOrCreate(
        impl->viCacheMutex, impl->vertexInputCache, viHash, [&]() -> Result<GplLibrary> {
            GplVertexInputBuilder vib(*impl->devicePtr);
            for (const auto& vb : vertBindings) {
                vib.vertexBinding(vb.binding, vb.stride, vb.inputRate);
            }
            for (const auto& va : vertAttributes) {
                vib.vertexAttribute(va.location, va.binding, va.format, va.offset);
            }
            vib.topology(topology);
            vib.cache(impl->cache);
            return vib.build();
        });
    if (!viResult.ok()) {
        destroyModules();
        if (ownsLayout)
            vkDestroyPipelineLayout(impl->device, pipelineLayout, nullptr);
        return std::move(viResult).error();
    }

    auto prResult =
        getOrCreate(impl->prCacheMutex, impl->preRasterCache, prHash, [&]() -> Result<GplLibrary> {
            GplPreRasterizationBuilder prb(*impl->devicePtr);
            prb.vertexModule(vertMod);
            prb.polygonMode(polygonMode);
            prb.cullMode(cullMode);
            prb.frontFace(frontFace);
            prb.pipelineLayout(pipelineLayout);
            prb.cache(impl->cache);
            for (auto ds : extraDynamicStates) {
                if (ds == VK_DYNAMIC_STATE_CULL_MODE || ds == VK_DYNAMIC_STATE_FRONT_FACE ||
                    ds == VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY) {
                    prb.dynamicState(ds);
                }
            }
            return prb.build();
        });
    if (!prResult.ok()) {
        destroyModules();
        if (ownsLayout)
            vkDestroyPipelineLayout(impl->device, pipelineLayout, nullptr);
        return std::move(prResult).error();
    }

    auto fsResult = getOrCreate(impl->fsCacheMutex, impl->fragmentShaderCache, fsHash,
                                [&]() -> Result<GplLibrary> {
                                    GplFragmentShaderBuilder fsb(*impl->devicePtr);
                                    fsb.fragmentModule(fragMod);
                                    if (depthFormat != VK_FORMAT_UNDEFINED) {
                                        fsb.depthTest(true, true, depthCompareOp);
                                    }
                                    fsb.pipelineLayout(pipelineLayout);
                                    fsb.cache(impl->cache);
                                    for (auto ds : extraDynamicStates) {
                                        if (ds == VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE ||
                                            ds == VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE ||
                                            ds == VK_DYNAMIC_STATE_DEPTH_COMPARE_OP) {
                                            fsb.dynamicState(ds);
                                        }
                                    }
                                    return fsb.build();
                                });
    if (!fsResult.ok()) {
        destroyModules();
        if (ownsLayout)
            vkDestroyPipelineLayout(impl->device, pipelineLayout, nullptr);
        return std::move(fsResult).error();
    }

    auto foResult = getOrCreate(impl->foCacheMutex, impl->fragmentOutputCache, foHash,
                                [&]() -> Result<GplLibrary> {
                                    GplFragmentOutputBuilder fob(*impl->devicePtr);
                                    fob.colorFormat(colorFormat);
                                    if (depthFormat != VK_FORMAT_UNDEFINED) {
                                        fob.depthFormat(depthFormat);
                                    }
                                    fob.samples(samples);
                                    if (enableBlending)
                                        fob.enableBlending();
                                    fob.cache(impl->cache);
                                    return fob.build();
                                });
    if (!foResult.ok()) {
        destroyModules();
        if (ownsLayout)
            vkDestroyPipelineLayout(impl->device, pipelineLayout, nullptr);
        return std::move(foResult).error();
    }

    auto viLib = viResult.value();
    auto prLib = prResult.value();
    auto fsLib = fsResult.value();
    auto foLib = foResult.value();

    // Fast-link (no optimization).
    auto linkResult = linkGplPipeline(*impl->devicePtr, *viLib, *prLib, *fsLib, *foLib,
                                      pipelineLayout, impl->cache, false);
    if (!linkResult.ok()) {
        destroyModules();
        if (ownsLayout)
            vkDestroyPipelineLayout(impl->device, pipelineLayout, nullptr);
        return std::move(linkResult).error();
    }

    VkPipeline fastLinked = linkResult.value();

    auto* handleImpl = new detail::PipelineHandleImpl;
    handleImpl->device = impl->device;
    handleImpl->baseline = fastLinked;
    handleImpl->optimized.store(VK_NULL_HANDLE, std::memory_order_relaxed);
    handleImpl->layout = pipelineLayout;
    handleImpl->ownsLayout = ownsLayout;
    handleImpl->bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

    PipelineHandle handle;
    handle.impl_ = handleImpl;

    // Step 3: Queue background optimization.
    auto* rawHandle = handleImpl;
    VkDevice capturedDevice = impl->device;
    const Device* capturedDevicePtr = impl->devicePtr;
    VkPipelineCache capturedCache = impl->cache;

    impl->enqueue({[=]() {
        // Check if handle was destroyed before we got scheduled.
        if (rawHandle->destroyed.load(std::memory_order_acquire))
            return;
        auto optResult = linkGplPipeline(*capturedDevicePtr, *viLib, *prLib, *fsLib, *foLib,
                                         rawHandle->layout, capturedCache, true);
        if (optResult.ok()) {
            // Try to publish the optimized pipeline. CAS ensures we don't
            // store into a handle that destroy() already exchanged away.
            VkPipeline expected = VK_NULL_HANDLE;
            if (!rawHandle->optimized.compare_exchange_strong(expected, optResult.value(),
                                                              std::memory_order_acq_rel))
                vkDestroyPipeline(capturedDevice, optResult.value(), nullptr);
        } else {
#ifndef NDEBUG
            std::fprintf(stderr, "[vksdl] background GPL optimization failed (non-fatal): %s\n",
                         optResult.error().message.c_str());
#endif
        }
    }});

    destroyModules();

    return handle;
}

void PipelineCompiler::waitIdle() {
    if (!impl_)
        return;
    auto* impl = static_cast<detail::PipelineCompilerImpl*>(impl_);
    while (impl->pending.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
    }
}

std::uint32_t PipelineCompiler::pendingCount() const {
    if (!impl_)
        return 0;
    auto* impl = static_cast<detail::PipelineCompilerImpl*>(impl_);
    return impl->pending.load(std::memory_order_acquire);
}

PipelineModel PipelineCompiler::resolvedModel() const {
    if (!impl_)
        return PipelineModel::Monolithic;
    return static_cast<detail::PipelineCompilerImpl*>(impl_)->resolvedModel;
}

PipelinePolicy PipelineCompiler::policy() const {
    if (!impl_)
        return PipelinePolicy::Auto;
    return static_cast<detail::PipelineCompilerImpl*>(impl_)->policy;
}

PipelineModelInfo PipelineCompiler::modelInfo() const {
    if (!impl_)
        return {};
    return static_cast<detail::PipelineCompilerImpl*>(impl_)->info;
}

} // namespace vksdl
