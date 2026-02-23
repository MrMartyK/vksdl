// Pipeline compiler demo: "material bomb."
//
// 64 cubes on a grid. As the camera orbits, crossing a sector boundary
// queues 16 pipeline permutations (128 total across 8 sectors), spawned
// 2 per frame to amortize compilation cost.
// Six modes toggled by keys 1-6:
//   1 -- Mono Cold     (fresh empty cache PER PIPELINE + synchronous build = worst case)
//   2 -- Mono Warm     (disk cache + synchronous build = cache-only benefit)
//   3 -- Auto Cold     (empty cache + GPL fast-link + async = GPL benefit alone)
//   4 -- Auto Warm     (disk cache + GPL fast-link + async = full benefit)
//   5 -- Auto Warm + stats overlay (dots: gray->yellow->green)
//   6 -- No-Surprises  (warm cache + FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT = prove
//                        the render thread never compiles)
//
// CLI flags for cross-process comparison:
//   --cold   Run only cold modes, save cache, exit.
//   --warm   Run only warm modes (requires cache from --cold run), exit.
//   --cycle  Run all 6 modes sequentially, exit.
//
// Frame time graph at the bottom auto-scales Y-axis to peak frame time.
// The contrast shows why PipelineCompiler matters: cold monolithic compilation
// causes 5-50ms spikes per pipeline, while warm cache / GPL returns in <1ms.

#include <vksdl/vksdl.hpp>

#include <SDL3/SDL.h>
#include <SDL3/SDL_scancode.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr int GRID            = 8;
static constexpr int CUBE_COUNT      = GRID * GRID;
static constexpr int SHADER_VARIANTS = 4;  // pbr_noise, voronoi, marble, iridescent
static constexpr int CULL_MODES      = 2;  // none, back
static constexpr int BLEND_MODES     = 2;  // opaque, alpha
static constexpr int POLY_MODES      = 2;  // fill, wireframe
static constexpr int DEPTH_CMP_MODES = 2;  // LESS_OR_EQUAL, LESS
static constexpr int PUSH_MODES      = 2;  // vertex-only, vertex+fragment
static constexpr int TOTAL_PERMS     = SHADER_VARIANTS * CULL_MODES * BLEND_MODES
                                     * POLY_MODES * DEPTH_CMP_MODES * PUSH_MODES; // 128
static constexpr int SECTOR_COUNT    = 8;  // camera orbit divided into 8 sectors
static constexpr int PERMS_PER_SECTOR = TOTAL_PERMS / SECTOR_COUNT; // 16
static constexpr int GRAPH_SAMPLES   = 300;
static constexpr float GRAPH_MIN_SCALE_MS = 5.0f;  // never scale below 5ms
static constexpr float CUBE_SPACING  = 1.8f;

struct Push {
    float mvp[16];
    float color[4];
};

struct MaterialDef {
    int  shader;   // 0=flat, 1=emissive, 2=checker, 3=stripe
    int  cull;     // 0=none, 1=back
    bool blend;    // false=opaque, true=alpha
    int  poly;     // 0=fill, 1=wireframe
    int  depthCmp; // 0=LESS_OR_EQUAL, 1=LESS
    int  push;     // 0=vertex-only, 1=vertex+fragment
};

static const char* shaderName(int s) {
    static const char* names[] = {"pbr_noise", "voronoi", "marble", "iridescent"};
    return names[s];
}

static const char* cullName(int c) {
    static const char* names[] = {"none", "back"};
    return names[c];
}

static const char* polyName(int p) {
    static const char* names[] = {"fill", "wire"};
    return names[p];
}

static const char* depthCmpName(int d) {
    static const char* names[] = {"LEQ", "LT"};
    return names[d];
}

static constexpr float cubePositions[] = {
    // Front (+Z)
    -0.5f, -0.5f,  0.5f,   0.5f, -0.5f,  0.5f,   0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,  -0.5f,  0.5f,  0.5f,  -0.5f, -0.5f,  0.5f,
    // Back (-Z)
    -0.5f, -0.5f, -0.5f,   0.5f,  0.5f, -0.5f,   0.5f, -0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,  -0.5f, -0.5f, -0.5f,  -0.5f,  0.5f, -0.5f,
    // Top (+Y)
    -0.5f,  0.5f, -0.5f,   0.5f,  0.5f,  0.5f,   0.5f,  0.5f, -0.5f,
     0.5f,  0.5f,  0.5f,  -0.5f,  0.5f, -0.5f,  -0.5f,  0.5f,  0.5f,
    // Bottom (-Y)
    -0.5f, -0.5f, -0.5f,   0.5f, -0.5f, -0.5f,   0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,  -0.5f, -0.5f,  0.5f,  -0.5f, -0.5f, -0.5f,
    // Right (+X)
     0.5f, -0.5f, -0.5f,   0.5f,  0.5f, -0.5f,   0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,   0.5f, -0.5f,  0.5f,   0.5f, -0.5f, -0.5f,
    // Left (-X)
    -0.5f, -0.5f, -0.5f,  -0.5f,  0.5f,  0.5f,  -0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f,  0.5f,  -0.5f, -0.5f, -0.5f,  -0.5f, -0.5f,  0.5f,
};
static constexpr std::uint32_t CUBE_VERTS = 36;

struct OverlayVertex {
    float x, y;
    float r, g, b, a;
};

// Push a quad (two triangles) into the overlay vertex list.
static void pushQuad(std::vector<OverlayVertex>& v,
                     float x0, float y0, float x1, float y1,
                     float r, float g, float b, float a = 1.0f) {
    v.push_back({x0, y0, r, g, b, a});
    v.push_back({x1, y0, r, g, b, a});
    v.push_back({x1, y1, r, g, b, a});
    v.push_back({x1, y1, r, g, b, a});
    v.push_back({x0, y1, r, g, b, a});
    v.push_back({x0, y0, r, g, b, a});
}

static vksdl::Mat4 mat4Translate(float x, float y, float z) {
    vksdl::Mat4 out = vksdl::mat4Identity();
    out.at(0, 3) = x;
    out.at(1, 3) = y;
    out.at(2, 3) = z;
    return out;
}

static vksdl::Mat4 mat4RotateY(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    vksdl::Mat4 out = vksdl::mat4Identity();
    out.at(0, 0) =  c;
    out.at(0, 2) =  s;
    out.at(2, 0) = -s;
    out.at(2, 2) =  c;
    return out;
}

// HSV (h in [0,1), s/v in [0,1]) to linear RGB.
static void hsvToRgb(float h, float s, float v, float out[3]) {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r, g, b;
    if      (h < 1.0f / 6.0f) { r = c; g = x; b = 0; }
    else if (h < 2.0f / 6.0f) { r = x; g = c; b = 0; }
    else if (h < 3.0f / 6.0f) { r = 0; g = c; b = x; }
    else if (h < 4.0f / 6.0f) { r = 0; g = x; b = c; }
    else if (h < 5.0f / 6.0f) { r = x; g = 0; b = c; }
    else                       { r = c; g = 0; b = x; }
    out[0] = r + m; out[1] = g + m; out[2] = b + m;
}

enum class Mode { MonoCold, MonoWarm, AutoCold, AutoWarm, AutoWarmStats, NoSurprises };

static const char* modeName(Mode m) {
    switch (m) {
    case Mode::MonoCold:      return "Mono (cold)";
    case Mode::MonoWarm:      return "Mono (warm)";
    case Mode::AutoCold:      return "Auto (cold)";
    case Mode::AutoWarm:      return "Auto (warm)";
    case Mode::AutoWarmStats: return "Auto (warm)+Stats";
    case Mode::NoSurprises:   return "No-Surprises (warm)";
    }
    return "?";
}

static bool isMono(Mode m) { return m == Mode::MonoCold || m == Mode::MonoWarm; }
static bool isAuto(Mode m) { return m == Mode::AutoCold || m == Mode::AutoWarm || m == Mode::AutoWarmStats; }

int main(int argc, char* argv[]) {
    // CLI flags:
    //   --cycle  Auto-advance through all 6 modes, exit after last one completes.
    //   --cold   Only run cold modes (MonoCold, AutoCold), save cache, exit.
    //            Use this as the first step of cross-process comparison.
    //   --warm   Only run warm modes (MonoWarm, AutoWarm, AutoWarmStats, NoSurprises).
    //            Requires a cache file from a previous --cold run.
    //
    // Cross-process workflow for honest warm vs cold comparison:
    //   del pipeline_compiler.cache
    //   pipeline_compiler.exe --cold     # builds cache, exits
    //   pipeline_compiler.exe --warm     # uses cache from previous run, exits
    enum class CycleMode { None, All, ColdOnly, WarmOnly };
    CycleMode cycleMode = CycleMode::None;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--cycle") cycleMode = CycleMode::All;
        else if (arg == "--cold") cycleMode = CycleMode::ColdOnly;
        else if (arg == "--warm") cycleMode = CycleMode::WarmOnly;
    }
    bool autoCycle = (cycleMode != CycleMode::None);

    auto app    = vksdl::App::create().value();
    auto window = app.createWindow("Pipeline Compiler", 1280, 720).value();

    auto instance = vksdl::InstanceBuilder{}
        .appName("vksdl_pipeline_compiler")
        .requireVulkan(1, 3)
        .enableWindowSupport()
        .build().value();

    auto surface = vksdl::Surface::create(instance, window).value();

    // GPL is detected opportunistically by DeviceBuilder (not required via needGPL()
    // because the demo must work on GPUs without GPL, falling back to monolithic).
    auto device = vksdl::DeviceBuilder(instance, surface)
        .needSwapchain()
        .needDynamicRendering()
        .needSync2()
        .preferDiscreteGpu()
        .requireCoreFeatures([](VkPhysicalDeviceFeatures& f) {
            f.fillModeNonSolid = VK_TRUE;
        })
        .build().value();

    auto swapchain = vksdl::SwapchainBuilder(device, surface)
        .size(window.pixelSize())
        .build().value();

    auto frames    = vksdl::FrameSync::create(device, swapchain.imageCount()).value();
    auto allocator = vksdl::Allocator::create(instance, device).value();

    std::printf("GPU: %s\n", device.gpuName());
    std::printf("GPL: %s | Fast-link: %s | PCCC: %s\n",
                device.hasGPL() ? "yes" : "no",
                device.hasGplFastLinking() ? "yes" : "no",
                device.hasPipelineCreationCacheControl() ? "yes" : "no");
    if (!device.hasGPL()) {
        std::printf("WARNING: GPU does not support VK_EXT_graphics_pipeline_library.\n"
                    "  Auto modes will fall back to monolithic compilation.\n");
    }

    auto makeDepth = [&]() {
        return vksdl::ImageBuilder(allocator)
            .size(swapchain.extent().width, swapchain.extent().height)
            .depthAttachment()
            .build().value();
    };
    auto depthImage = makeDepth();

    std::filesystem::path cachePath = vksdl::exeDir() / "pipeline_compiler.cache";
    bool haveDiskCache = std::filesystem::exists(cachePath);
    auto warmCache = vksdl::PipelineCache::load(device, cachePath).value();
    std::printf("Warm cache: %s (%zu bytes)\n",
                haveDiskCache ? "loaded" : "empty (first run)", warmCache.dataSize());

    // Cold cache starts empty -- recreated on each mode-1 reset.
    auto coldCache = vksdl::PipelineCache::create(device).value();

    std::filesystem::path shaderDir = vksdl::exeDir() / "shaders";
    const char* fragFiles[] = {
        "pbr_noise.frag.spv", "voronoi.frag.spv", "marble.frag.spv", "iridescent.frag.spv"
    };

    auto cubeVB = vksdl::uploadVertexBuffer(
        allocator, device, cubePositions, sizeof(cubePositions)).value();

    static constexpr std::size_t MAX_OVERLAY_VERTS = 4096;
    std::vector<vksdl::Buffer> overlayVBs;
    overlayVBs.reserve(swapchain.imageCount());
    for (std::uint32_t fi = 0; fi < swapchain.imageCount(); ++fi) {
        overlayVBs.push_back(vksdl::BufferBuilder(allocator)
            .usage(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
            .size(sizeof(OverlayVertex) * MAX_OVERLAY_VERTS)
            .mapped()
            .build().value());
    }

    auto overlayPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "overlay.vert.spv")
        .fragmentShader(shaderDir / "overlay.frag.spv")
        .colorFormat(swapchain)
        .vertexBinding(0, sizeof(OverlayVertex))
        .vertexAttribute(0, 0, VK_FORMAT_R32G32_SFLOAT, 0)
        .vertexAttribute(1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(OverlayVertex, r))
        .enableBlending()
        .build().value();

    // Built with the warm cache so it doesn't stutter at startup.
    auto defaultPipeline = vksdl::PipelineBuilder(device)
        .vertexShader(shaderDir / "scene.vert.spv")
        .fragmentShader(shaderDir / "pbr_noise.frag.spv")
        .colorFormat(swapchain)
        .depthFormat(VK_FORMAT_D32_SFLOAT)
        .vertexBinding(0, sizeof(float) * 3)
        .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .pushConstants<Push>(VK_SHADER_STAGE_VERTEX_BIT)
        .cache(warmCache)
        .build().value();

    auto makeBuilder = [&](const MaterialDef& mat, vksdl::PipelineCache& pipeCache) {
        VkShaderStageFlags pushStages = VK_SHADER_STAGE_VERTEX_BIT;
        if (mat.push == 1)
            pushStages |= VK_SHADER_STAGE_FRAGMENT_BIT;

        auto builder = vksdl::PipelineBuilder(device)
            .vertexShader(shaderDir / "scene.vert.spv")
            .fragmentShader(shaderDir / fragFiles[mat.shader])
            .colorFormat(swapchain)
            .depthFormat(VK_FORMAT_D32_SFLOAT)
            .vertexBinding(0, sizeof(float) * 3)
            .vertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .pushConstantRange({pushStages, 0, sizeof(Push)})
            .cache(pipeCache);

        if (mat.cull == 1)     builder.cullBack();
        if (mat.blend)         builder.enableBlending();
        if (mat.poly == 1)     builder.wireframe();
        if (mat.depthCmp == 1) builder.depthCompareOp(VK_COMPARE_OP_LESS);

        return builder;
    };

    std::array<MaterialDef, TOTAL_PERMS> perms{};
    {
        int i = 0;
        for (int s = 0; s < SHADER_VARIANTS; ++s)
            for (int c = 0; c < CULL_MODES; ++c)
                for (int b = 0; b < BLEND_MODES; ++b)
                    for (int p = 0; p < POLY_MODES; ++p)
                        for (int d = 0; d < DEPTH_CMP_MODES; ++d)
                            for (int pc = 0; pc < PUSH_MODES; ++pc)
                                perms[static_cast<std::size_t>(i++)] = {s, c, b != 0, p, d, pc};

        // Deterministic shuffle (simple LCG seed).
        std::uint32_t seed = 42;
        for (int j = TOTAL_PERMS - 1; j > 0; --j) {
            seed = seed * 1664525u + 1013904223u;
            int k = static_cast<int>(seed % static_cast<std::uint32_t>(j + 1));
            std::swap(perms[static_cast<std::size_t>(j)],
                      perms[static_cast<std::size_t>(k)]);
        }
    }

    struct CubeData {
        float x, z;
        float hue;
        int   materialIndex; // index into perms[], -1 = default
    };

    std::array<CubeData, CUBE_COUNT> cubes{};
    {
        float half = (GRID - 1) * CUBE_SPACING * 0.5f;
        for (int i = 0; i < CUBE_COUNT; ++i) {
            int row = i / GRID;
            int col = i % GRID;
            auto& c = cubes[static_cast<std::size_t>(i)];
            c.x = col * CUBE_SPACING - half;
            c.z = row * CUBE_SPACING - half;
            c.hue = static_cast<float>(i) / static_cast<float>(CUBE_COUNT);
            // Each cube gets a permutation via modular wrap (128 perms across 64 cubes
            // means each perm maps to exactly one cube via p % CUBE_COUNT, and each
            // cube gets the highest-indexed perm that maps to it).
            c.materialIndex = -1;
        }
        // Assign permutations to cubes deterministically: each perm gets exactly
        // one cube. With 128 perms and 64 cubes, each cube shows 2 permutations
        // (the last one assigned wins for rendering, but both get spawned).
        for (int p = 0; p < TOTAL_PERMS; ++p) {
            cubes[static_cast<std::size_t>(p % CUBE_COUNT)].materialIndex = p;
        }
    }

    Mode mode = Mode::MonoCold;
    int  spawnQueued = 0;   // how many perms have been queued by sector crossings
    int  spawnNext   = 0;  // index of next perm to spawn (also = count of spawned)
    int  lastSector  = -1;
    static constexpr int SPAWN_PER_FRAME = 2; // amortized: max 2 pipeline creates per frame

    // Pipeline storage (indexed by permutation index).
    std::unordered_map<int, vksdl::Pipeline>       monoPipelines;
    std::unordered_map<int, vksdl::PipelineHandle> asyncPipelines;
    std::optional<vksdl::PipelineCompiler>          compiler;

    // Track which permutations have been spawned in current mode.
    std::array<bool, TOTAL_PERMS> spawned{};

    // Per-spawn data (for statistics).
    struct SpawnRecord {
        int    permIdx;
        double ms;           // time on render thread (cache probe + fast-link or full build)
        bool   cacheHit;     // from VkPipelineCreationFeedback (mono) or inferred (auto)
        bool   cacheProbeHit; // true if FAIL_ON_PIPELINE_COMPILE_REQUIRED succeeded (auto only)
    };
    std::vector<SpawnRecord> spawnRecords;
    spawnRecords.reserve(TOTAL_PERMS);
    bool statsPrinted = false;
    int  noSurprisesMisses = 0; // pipelines that would have required compilation

    // Frame time graph ring buffer.
    std::array<float, GRAPH_SAMPLES> graphTimes{};
    int graphIndex = 0;

    // Print spawn statistics after all permutations are done.
    auto printStats = [&]() {
        if (spawnRecords.empty()) return;

        int n = static_cast<int>(spawnRecords.size());

        // Sorted times for percentile computation.
        std::vector<double> sorted;
        sorted.reserve(static_cast<std::size_t>(n));
        for (const auto& r : spawnRecords) sorted.push_back(r.ms);
        std::sort(sorted.begin(), sorted.end());

        double total = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        double mean  = total / static_cast<double>(n);
        double median = (n % 2 == 0)
            ? (sorted[static_cast<std::size_t>(n / 2 - 1)] + sorted[static_cast<std::size_t>(n / 2)]) / 2.0
            : sorted[static_cast<std::size_t>(n / 2)];
        double p90 = sorted[static_cast<std::size_t>(std::min(n * 90 / 100, n - 1))];
        double p95 = sorted[static_cast<std::size_t>(std::min(n * 95 / 100, n - 1))];
        double p99 = sorted[static_cast<std::size_t>(std::min(n * 99 / 100, n - 1))];

        // Standard deviation.
        double variance = 0.0;
        for (double t : sorted) variance += (t - mean) * (t - mean);
        double stddev = std::sqrt(variance / static_cast<double>(n));

        // Threshold counts.
        int above1 = 0, above5 = 0, above16 = 0;
        for (double t : sorted) {
            if (t > 1.0)  ++above1;
            if (t > 5.0)  ++above5;
            if (t > 16.0) ++above16;
        }

        // Cache hit count.
        int cacheHits = 0;
        for (const auto& r : spawnRecords) {
            if (r.cacheHit) ++cacheHits;
        }

        // Top-5 slowest (sort records by time descending).
        std::vector<SpawnRecord> bySlowest = spawnRecords;
        std::sort(bySlowest.begin(), bySlowest.end(),
                  [](const SpawnRecord& a, const SpawnRecord& b) { return a.ms > b.ms; });

        bool cold = (mode == Mode::MonoCold || mode == Mode::AutoCold);

        std::printf("\n%s: %d pipelines\n", modeName(mode), n);
        if (mode == Mode::MonoCold) {
            std::printf("  Cache:  fresh empty VkPipelineCache per pipeline (no cross-pipeline reuse)\n");
        } else {
            std::printf("  Cache:  %s\n", cold ? "empty VkPipelineCache (shared across batch)"
                                               : "warm (loaded from disk)");
        }
        const char* strategy = isMono(mode) ? "synchronous monolithic"
                             : isAuto(mode) ? "cache probe -> GPL fast-link -> async optimize"
                             : "FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT (cache-only, zero compilation)";
        std::printf("  Strategy: %s\n", strategy);
        if (isAuto(mode) && compiler) {
            auto info = compiler->modelInfo();
            std::printf("  Resolved: %s | PCCC: %s | GPL: %s | FastLink: %s\n",
                        info.model == vksdl::PipelineModel::GPL ? "GPL" : "Mono",
                        info.hasPCCC ? "yes" : "no",
                        info.hasGPL ? "yes" : "no",
                        info.fastLink ? "yes" : "no");
        }
        int cacheMisses = n - cacheHits;

        std::printf("  ----\n");
        std::printf("  All %d pipelines:\n", n);
        std::printf("    Total:  %.1f ms\n", total);
        std::printf("    Mean:   %.2f ms  (stddev: %.2f ms)\n", mean, stddev);
        std::printf("    Median: %.2f ms\n", median);
        std::printf("    Min:    %.2f ms | Max: %.2f ms\n", sorted.front(), sorted.back());
        std::printf("    P90:    %.2f ms | P95: %.2f ms | P99: %.2f ms\n", p90, p95, p99);
        std::printf("    >1ms: %d | >5ms: %d | >16ms: %d (frame-drop)\n",
                    above1, above5, above16);
        const char* hitSource = isMono(mode) ? "VkPipelineCreationFeedback"
                             : mode == Mode::NoSurprises ? "cache probe result"
                             : "cache probe or timing heuristic (<0.5ms)";
        std::printf("  Cache: %d hits, %d misses (%s)\n",
                    cacheHits, cacheMisses, hitSource);

        // Miss-only stats (misses cause stutter; hits don't).
        if (cacheMisses > 0) {
            std::vector<double> missTimes;
            for (const auto& r : spawnRecords) {
                if (!r.cacheHit) missTimes.push_back(r.ms);
            }
            std::sort(missTimes.begin(), missTimes.end());
            int nm = static_cast<int>(missTimes.size());
            double missTotal = std::accumulate(missTimes.begin(), missTimes.end(), 0.0);
            double missMean  = missTotal / static_cast<double>(nm);
            double missP95   = missTimes[static_cast<std::size_t>(std::min(nm * 95 / 100, nm - 1))];
            double missP99   = missTimes[static_cast<std::size_t>(std::min(nm * 99 / 100, nm - 1))];
            std::printf("  Misses only (%d): mean %.2f ms | p95 %.2f ms | p99 %.2f ms | max %.2f ms\n",
                        nm, missMean, missP95, missP99, missTimes.back());
        }

        if (isAuto(mode)) {
            int optimized = 0;
            for (auto& [k, h] : asyncPipelines) {
                if (h.isOptimized()) ++optimized;
            }
            // Count cache probe hits vs GPL fast-links.
            int probeHits = 0;
            double probeTotal = 0.0, fastLinkTotal = 0.0;
            int fastLinkCount = 0;
            for (const auto& r : spawnRecords) {
                if (r.cacheProbeHit) {
                    ++probeHits;
                    probeTotal += r.ms;
                } else {
                    ++fastLinkCount;
                    fastLinkTotal += r.ms;
                }
            }
            std::printf("  ----\n");
            std::printf("  Acquisition breakdown:\n");
            std::printf("    Cache probe hits:  %d/%d", probeHits, n);
            if (probeHits > 0)
                std::printf(" (mean %.2f ms -- fully optimized, zero render-thread compilation)",
                            probeTotal / static_cast<double>(probeHits));
            std::printf("\n");
            std::printf("    GPL fast-links:    %d/%d", fastLinkCount, n);
            if (fastLinkCount > 0)
                std::printf(" (mean %.2f ms -- usable immediately, optimized in background)",
                            fastLinkTotal / static_cast<double>(fastLinkCount));
            std::printf("\n");
            std::printf("  Background optimized: %d/%d\n", optimized, n);
            std::printf("  Time to usable (T_usable):  render-thread cost shown above\n");
            std::printf("  Time to optimized (T_opt):  asynchronous, off render thread\n");
        }

        // Top-5 slowest.
        int top = std::min(5, n);
        std::printf("  Top %d slowest:\n", top);
        for (int i = 0; i < top; ++i) {
            const auto& r = bySlowest[static_cast<std::size_t>(i)];
            const auto& mat = perms[static_cast<std::size_t>(r.permIdx)];
            std::printf("    #%d  [%3d] %s/%s/%s/%s: %.2f ms %s\n",
                        i + 1, r.permIdx,
                        shaderName(mat.shader), cullName(mat.cull),
                        mat.blend ? "alpha" : "opaque", polyName(mat.poly),
                        r.ms, r.cacheHit ? "(cache hit)" : "");
        }

        if (mode == Mode::NoSurprises) {
            int probeSuccesses = 0;
            int probeFails = 0;
            for (const auto& r : spawnRecords) {
                if (r.cacheProbeHit) ++probeSuccesses;
                else ++probeFails;
            }
            std::printf("  ----\n");
            std::printf("  No-surprises verdict:\n");
            std::printf("    Cache probe successes: %d/%d (zero compilation on render thread)\n",
                        probeSuccesses, n);
            std::printf("    Cache probe failures:  %d/%d (would have required compilation)\n",
                        probeFails, n);
            if (probeFails == 0) {
                std::printf("    PASS: warm cache contains every permutation. "
                            "The render thread never compiles.\n");
            } else {
                std::printf("    FAIL: %d permutations missing from cache. "
                            "Run a cold mode first, quit, re-launch.\n", probeFails);
            }
        }

        if (cold) {
            std::printf("  Note: empty VkPipelineCache does not guarantee cold compilation.\n"
                        "        Internal driver caches (e.g. NVIDIA GLCache) may persist across\n"
                        "        VkPipelineCache objects, runs, and even driver versions.\n"
                        "        For true cold numbers, clear driver shader cache and restart.\n");
        }
        std::printf("\n");
    };

    // Reset all mode state (called on mode switch).
    auto resetMode = [&](Mode newMode) {
        // Drain background compilations before snapshotting stats.
        if (compiler) compiler->waitIdle();

        // Print statistics from previous mode before clearing.
        if (!statsPrinted && !spawnRecords.empty()) {
            printStats();
        }

        device.waitIdle();

        // Compiler must be destroyed before handles (bg threads reference handles).
        compiler.reset();
        monoPipelines.clear();
        asyncPipelines.clear();
        spawned.fill(false);
        spawnRecords.clear();
        statsPrinted = false;
        noSurprisesMisses = 0;
        spawnQueued = 0;
        spawnNext   = 0;
        lastSector  = -1;
        graphTimes.fill(0.0f);
        graphIndex = 0;

        // Each mode gets a fresh cache to isolate variables.
        // Cold modes: empty cache. Warm modes: reload from disk.
        bool cold = (newMode == Mode::MonoCold || newMode == Mode::AutoCold);

        if (cold) {
            coldCache = vksdl::PipelineCache::create(device).value();
            std::printf("Cache: empty (cold)\n");
        } else {
            // Reload from disk each time so previous mode's in-memory
            // additions don't leak across. This also means NVIDIA's
            // VkPipelineCache deserialization cost is included honestly.
            using Clock = std::chrono::high_resolution_clock;
            auto loadStart = Clock::now();
            warmCache = vksdl::PipelineCache::load(device, cachePath).value();
            auto loadEnd = Clock::now();
            double loadMs = std::chrono::duration<double, std::milli>(loadEnd - loadStart).count();
            std::printf("Cache: %zu bytes (warm from disk, loaded in %.1fms)\n",
                        warmCache.dataSize(), loadMs);
            if (warmCache.dataSize() < 100) {
                std::printf("  WARNING: warm cache is effectively empty (header only).\n"
                            "  Run a cold mode first, quit, then re-launch for meaningful warm results.\n");
            }
        }

        if (isAuto(newMode)) {
            auto& cacheRef = cold ? coldCache : warmCache;
            auto cr = vksdl::PipelineCompiler::create(
                device, cacheRef, vksdl::PipelinePolicy::Auto);
            if (cr.ok()) {
                compiler.emplace(std::move(cr.value()));
                std::printf("Compiler: resolved to %s\n",
                            compiler->resolvedModel() == vksdl::PipelineModel::GPL
                                ? "GPL" : "Monolithic");
            } else {
                std::printf("Compiler creation failed, falling back to Mono\n");
                newMode = cold ? Mode::MonoCold : Mode::MonoWarm;
            }
        }

        if (newMode == Mode::NoSurprises) {
            std::printf("Strategy: FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT on every pipeline.\n"
                        "          If cache is warm, every pipeline should succeed with zero compilation.\n"
                        "          Any miss proves the cache is incomplete.\n");
        }

        mode = newMode;
        std::printf("\nMode: %s\n", modeName(mode));
    };

    // Spawn a single material permutation in current mode.
    auto spawnMaterial = [&](int permIdx) {
        using Clock = std::chrono::high_resolution_clock;
        const auto& mat = perms[static_cast<std::size_t>(permIdx)];
        bool cold = (mode == Mode::MonoCold || mode == Mode::AutoCold);
        auto& cacheRef = cold ? coldCache : warmCache;

        if (mode == Mode::NoSurprises) {
            // No-surprises mode: every pipeline uses FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT.
            // If the warm cache contains this pipeline, creation succeeds with zero
            // compilation. If not, creation fails with VK_PIPELINE_COMPILE_REQUIRED
            // and we fall back to the default pipeline.
            auto builder = makeBuilder(mat, warmCache);
            auto t0 = Clock::now();
            auto result = builder.buildWithFlags(
                VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT);
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (result.ok()) {
                // Cache probe succeeded: the pipeline was in the cache by definition.
                // VkPipelineCreationFeedback may still report false due to driver
                // quirks, but the probe success is the authoritative signal.
                monoPipelines.emplace(permIdx, std::move(result.value()));
                spawned[static_cast<std::size_t>(permIdx)] = true;
                spawnRecords.push_back({permIdx, ms, true, true});
                std::printf("[nosurp %3d] %s/%s/%s/%s/%s: %.2fms (cache hit, zero compilation)\n",
                            permIdx, shaderName(mat.shader), cullName(mat.cull),
                            mat.blend ? "alpha" : "opaque", polyName(mat.poly),
                            depthCmpName(mat.depthCmp), ms);
            } else {
                // VK_PIPELINE_COMPILE_REQUIRED -- cache miss, would need compilation.
                // Use default pipeline instead. This is the whole point: prove
                // the render thread never compiles.
                ++noSurprisesMisses;
                spawned[static_cast<std::size_t>(permIdx)] = true; // mark as handled
                spawnRecords.push_back({permIdx, ms, false, false});
                std::printf("[nosurp %3d] %s/%s/%s/%s/%s: %.2fms MISS (would require compilation)\n",
                            permIdx, shaderName(mat.shader), cullName(mat.cull),
                            mat.blend ? "alpha" : "opaque", polyName(mat.poly),
                            depthCmpName(mat.depthCmp), ms);
            }
            return;
        }

        if (isMono(mode)) {
            // MonoCold: fresh empty cache PER PIPELINE so no cross-pipeline
            // cache hits within the batch. Each pipeline pays full compilation
            // cost (modulo driver-internal caches we can't control).
            // MonoWarm: shared warm cache from disk (simulates engine steady state).
            std::optional<vksdl::PipelineCache> perPipeCache;
            if (mode == Mode::MonoCold)
                perPipeCache.emplace(vksdl::PipelineCache::create(device).value());
            auto& effectiveCache = perPipeCache ? *perPipeCache : cacheRef;

            auto builder = makeBuilder(mat, effectiveCache);
            auto t0 = Clock::now();
            auto result = builder.build();
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (result.ok()) {
                bool hit = false;
                if (auto* fb = result.value().feedback())
                    hit = fb->valid && fb->cacheHit;
                monoPipelines.emplace(permIdx, std::move(result.value()));
                spawned[static_cast<std::size_t>(permIdx)] = true;
                spawnRecords.push_back({permIdx, ms, hit, false});
                std::printf("[mono %3d] %s/%s/%s/%s/%s: %.1fms%s\n",
                            permIdx, shaderName(mat.shader), cullName(mat.cull),
                            mat.blend ? "alpha" : "opaque", polyName(mat.poly),
                            depthCmpName(mat.depthCmp), ms,
                            hit ? " (cache hit)" : "");

                // Merge per-pipeline cache data into warmCache so it can be
                // saved to disk at exit. Without this, cold mode compilations
                // would be lost (perPipeCache is destroyed each iteration).
                if (perPipeCache) {
                    VkPipelineCache src = perPipeCache->vkPipelineCache();
                    vkMergePipelineCaches(device.vkDevice(),
                                         warmCache.vkPipelineCache(), 1, &src);
                }
            } else {
                std::fprintf(stderr, "[mono %3d] BUILD FAILED: %s\n",
                             permIdx, result.error().message.c_str());
            }
        } else {
            auto builder = makeBuilder(mat, cacheRef);
            auto t0 = Clock::now();
            auto result = compiler->compile(builder);
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (result.ok()) {
                // Detect cache probe hit vs GPL fast-link.
                // isOptimized() is only meaningful as a probe indicator when
                // resolvedModel() == GPL. In Monolithic mode, isOptimized() is
                // always true (both probe hits and synchronous builds set it).
                bool isGplPath = compiler &&
                    compiler->resolvedModel() == vksdl::PipelineModel::GPL;
                bool probeHit = result.value().isOptimized() && isGplPath;
                // For monolithic fallback, use timing heuristic only.
                if (!isGplPath)
                    probeHit = (ms < 0.5);
                bool likelyCacheHit = probeHit || (ms < 0.5);
                const char* tag = probeHit    ? " (cache probe hit)"
                                : likelyCacheHit ? " (fast-link, likely cached parts)"
                                : " (fast-link, new parts)";
                asyncPipelines.emplace(permIdx, std::move(result.value()));
                spawned[static_cast<std::size_t>(permIdx)] = true;
                spawnRecords.push_back({permIdx, ms, likelyCacheHit, probeHit});
                std::printf("[auto %3d] %s/%s/%s/%s/%s: %.2fms%s\n",
                            permIdx, shaderName(mat.shader), cullName(mat.cull),
                            mat.blend ? "alpha" : "opaque", polyName(mat.poly),
                            depthCmpName(mat.depthCmp), ms, tag);
            } else {
                std::fprintf(stderr, "[auto %3d] COMPILE FAILED: %s\n",
                             permIdx, result.error().message.c_str());
            }
        }
    };

    using Clock = std::chrono::high_resolution_clock;
    bool running = true;
    vksdl::Event event;
    int frameNum = 0;
    auto startTime = Clock::now();
    auto lastTitleUpdate = Clock::now();
    float lastDtMs = 0.0f;

    // Set initial mode based on cycle mode.
    Mode initialMode = Mode::MonoCold;
    if (cycleMode == CycleMode::WarmOnly) {
        initialMode = Mode::MonoWarm;
    }

    if (autoCycle) {
        const char* cycleDesc = cycleMode == CycleMode::ColdOnly  ? "cold modes only (MonoCold, AutoCold)"
                              : cycleMode == CycleMode::WarmOnly  ? "warm modes only (MonoWarm, AutoWarm, AutoWarm+Stats, NoSurprises)"
                              : "all 6 modes";
        std::printf("Auto-cycle: %s\n", cycleDesc);
        if (cycleMode == CycleMode::WarmOnly && warmCache.dataSize() < 100) {
            std::printf("WARNING: --warm but cache is header-only (%zu bytes).\n"
                        "  Run with --cold first to build the cache file.\n",
                        warmCache.dataSize());
        }
        if (cycleMode == CycleMode::ColdOnly) {
            std::printf("Cross-process workflow:\n"
                        "  1. del pipeline_compiler.cache\n"
                        "  2. pipeline_compiler.exe --cold   (builds cache, exits)\n"
                        "  3. pipeline_compiler.exe --warm   (uses warm cache, exits)\n");
        }
        // Queue all permutations immediately in cycle mode.
        spawnQueued = TOTAL_PERMS;
    } else {
        std::printf("Controls: 1=Mono(cold) 2=Mono(warm) 3=Auto(cold) 4=Auto(warm) 5=Auto+Stats 6=NoSurprises ESC=quit\n");
        std::printf("Compare: 1 vs 3 (mono vs auto, both cold) | 2 vs 4 (mono vs auto, both warm) | 6 = prove zero compilation\n");
        std::printf("CLI: --cold (cold modes only) | --warm (warm modes only) | --cycle (all 6 modes)\n");
    }
    std::printf("Note: 'cold' = empty VkPipelineCache. Internal driver caches may persist across objects/runs.\n");

    // Apply initial mode (warm modes need cache reload).
    if (initialMode != Mode::MonoCold) {
        resetMode(initialMode);
        if (autoCycle) spawnQueued = TOTAL_PERMS;
    } else {
        std::printf("Mode: %s\n", modeName(mode));
    }

    while (running) {
        // Event handling (before render timing starts).
        while (window.pollEvent(event)) {
            if (event.type == vksdl::EventType::CloseRequested ||
                event.type == vksdl::EventType::Quit) {
                running = false;
            }
            if (event.type == vksdl::EventType::KeyDown) {
                if (event.key == SDL_SCANCODE_ESCAPE) running = false;
                else if (event.key == SDL_SCANCODE_1 && mode != Mode::MonoCold)
                    resetMode(Mode::MonoCold);
                else if (event.key == SDL_SCANCODE_2 && mode != Mode::MonoWarm)
                    resetMode(Mode::MonoWarm);
                else if (event.key == SDL_SCANCODE_3 && mode != Mode::AutoCold)
                    resetMode(Mode::AutoCold);
                else if (event.key == SDL_SCANCODE_4 && mode != Mode::AutoWarm)
                    resetMode(Mode::AutoWarm);
                else if (event.key == SDL_SCANCODE_5 && mode != Mode::AutoWarmStats)
                    resetMode(Mode::AutoWarmStats);
                else if (event.key == SDL_SCANCODE_6 && mode != Mode::NoSurprises)
                    resetMode(Mode::NoSurprises);
            }
        }

        if (window.consumeResize()) {
            (void)swapchain.recreate(device, window);
            depthImage = makeDepth();
        }

        // Acquire frame (includes fence wait -- excluded from render timing).
        auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();

        // Start render timing after acquire (excludes fence wait and event processing).
        auto renderStart = Clock::now();

        float elapsed = std::chrono::duration<float>(renderStart - startTime).count();

        // Camera: slow orbit.
        float camAngle = elapsed * 0.15f;

        // Sector-based spawning: divide orbit into 8 sectors (45 degrees each).
        // Crossing a sector boundary queues 16 permutations. Actual creation is
        // amortized at SPAWN_PER_FRAME per frame to avoid stutter even in mono mode
        // turning into a multi-hundred-ms freeze.
        float sectorAngle = std::numbers::pi_v<float> / 4.0f; // 45 degrees
        int currentSector = static_cast<int>(std::fmod(camAngle, 2.0f * std::numbers::pi_v<float>) / sectorAngle) % SECTOR_COUNT;
        if (currentSector != lastSector && spawnQueued < TOTAL_PERMS) {
            spawnQueued = std::min(spawnQueued + PERMS_PER_SECTOR, TOTAL_PERMS);
            lastSector = currentSector;
        }
        // Drain spawn queue: create up to N pipelines this frame.
        // In cycle mode, spawn all at once to finish quickly.
        int spawnLimit = autoCycle ? TOTAL_PERMS : SPAWN_PER_FRAME;
        for (int i = 0; i < spawnLimit && spawnNext < spawnQueued; ++i) {
            spawnMaterial(spawnNext);
            ++spawnNext;
        }
        // Print statistics once after all permutations are spawned.
        if (spawnNext >= TOTAL_PERMS && !statsPrinted) {
            if (isAuto(mode) && compiler) compiler->waitIdle();
            printStats();
            statsPrinted = true;

            // Auto-cycle: advance to next mode or exit.
            if (autoCycle) {
                // Build the mode sequence based on --cold / --warm / --cycle.
                std::vector<Mode> modeSeq;
                if (cycleMode == CycleMode::ColdOnly) {
                    modeSeq = {Mode::MonoCold, Mode::AutoCold};
                } else if (cycleMode == CycleMode::WarmOnly) {
                    modeSeq = {Mode::MonoWarm, Mode::AutoWarm,
                               Mode::AutoWarmStats, Mode::NoSurprises};
                } else {
                    modeSeq = {Mode::MonoCold, Mode::MonoWarm, Mode::AutoCold,
                               Mode::AutoWarm, Mode::AutoWarmStats, Mode::NoSurprises};
                }
                bool found = false;
                for (std::size_t m = 0; m + 1 < modeSeq.size(); ++m) {
                    if (mode == modeSeq[m]) {
                        resetMode(modeSeq[m + 1]);
                        spawnQueued = TOTAL_PERMS;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // Last mode in sequence, done.
                    running = false;
                }
            }
        }
        float camDist  = 14.0f;
        float eye[3]    = {camDist * std::sin(camAngle), 7.0f,
                           camDist * std::cos(camAngle)};
        float target[3] = {0.0f, 0.0f, 0.0f};
        float up[3]     = {0.0f, 1.0f, 0.0f};

        vksdl::Mat4 view = vksdl::lookAt(eye, target, up);

        float aspect = static_cast<float>(swapchain.extent().width)
                     / static_cast<float>(swapchain.extent().height);
        static constexpr float kFovY = 0.7854f; // 45 degrees in radians
        vksdl::Mat4 proj = vksdl::perspectiveForwardZ(kFovY, aspect, 0.1f, 100.0f);
        vksdl::Mat4 vp   = vksdl::mat4Mul(proj, view);

        VkCommandBuffer cmd = frame.cmd;
        vksdl::beginOneTimeCommands(cmd);

        vksdl::transitionToColorAttachment(cmd, img.image);
        vksdl::transitionToDepthAttachment(cmd, depthImage.vkImage());

        VkRenderingAttachmentInfo colorAttach{};
        colorAttach.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttach.imageView   = img.view;
        colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttach.clearValue.color = {{0.06f, 0.06f, 0.08f, 1.0f}};

        VkRenderingAttachmentInfo depthAttach{};
        depthAttach.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttach.imageView   = depthImage.vkImageView();
        depthAttach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttach.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttach.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo renderInfo{};
        renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea           = {{0, 0}, swapchain.extent()};
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttach;
        renderInfo.pDepthAttachment     = &depthAttach;

        vkCmdBeginRendering(cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width    = static_cast<float>(swapchain.extent().width);
        viewport.height   = static_cast<float>(swapchain.extent().height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapchain.extent()};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkDeviceSize vbOffset = 0;
        VkBuffer vb = cubeVB.vkBuffer();
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);

        // Draw each cube with the appropriate pipeline.
        VkPipeline         lastBound       = VK_NULL_HANDLE;
        VkPipelineLayout   lastBoundLayout = VK_NULL_HANDLE;
        VkShaderStageFlags lastPushStages  = VK_SHADER_STAGE_VERTEX_BIT;

        for (int i = 0; i < CUBE_COUNT; ++i) {
            const auto& cube = cubes[static_cast<std::size_t>(i)];

            // Compute model matrix: translate to grid position, spin slowly.
            float spinSpeed = 0.3f + 0.5f * cube.hue;
            vksdl::Mat4 model = vksdl::mat4Mul(
                mat4Translate(cube.x, 0.0f, cube.z),
                mat4RotateY(elapsed * spinSpeed));
            vksdl::Mat4 mvp = vksdl::mat4Mul(vp, model);

            // Cube color from hue.
            float rgb[3];
            hsvToRgb(cube.hue, 0.85f, 0.95f, rgb);

            int matIdx = cube.materialIndex;
            bool hasSpawned = (matIdx >= 0 && matIdx < TOTAL_PERMS
                               && spawned[static_cast<std::size_t>(matIdx)]);
            float alpha = 1.0f;
            if (hasSpawned)
                alpha = perms[static_cast<std::size_t>(matIdx)].blend ? 0.7f : 1.0f;

            // Determine which pipeline to bind.
            VkPipeline         pipeline   = VK_NULL_HANDLE;
            VkPipelineLayout   layout     = VK_NULL_HANDLE;
            VkShaderStageFlags pushStages = VK_SHADER_STAGE_VERTEX_BIT;

            if (hasSpawned) {
                const auto& mat = perms[static_cast<std::size_t>(matIdx)];
                pushStages = VK_SHADER_STAGE_VERTEX_BIT;
                if (mat.push == 1)
                    pushStages |= VK_SHADER_STAGE_FRAGMENT_BIT;

                if (isMono(mode) || mode == Mode::NoSurprises) {
                    auto it = monoPipelines.find(matIdx);
                    if (it != monoPipelines.end()) {
                        pipeline = it->second.vkPipeline();
                        layout   = it->second.vkPipelineLayout();
                    }
                } else {
                    auto it = asyncPipelines.find(matIdx);
                    if (it != asyncPipelines.end()) {
                        pipeline = it->second.vkPipeline();
                        layout   = it->second.vkPipelineLayout();
                    }
                }
            }

            // Fall back to default.
            if (pipeline == VK_NULL_HANDLE) {
                pipeline   = defaultPipeline.vkPipeline();
                layout     = defaultPipeline.vkPipelineLayout();
                pushStages = VK_SHADER_STAGE_VERTEX_BIT;
            }

            // Bind pipeline only when it changes.
            if (pipeline != lastBound) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                lastBound       = pipeline;
                lastBoundLayout = layout;
                lastPushStages  = pushStages;
            }

            Push push{};
            std::memcpy(push.mvp, mvp.data(), sizeof(push.mvp));
            push.color[0] = rgb[0];
            push.color[1] = rgb[1];
            push.color[2] = rgb[2];
            push.color[3] = alpha;

            vkCmdPushConstants(cmd, lastBoundLayout,
                               lastPushStages,
                               0, sizeof(Push), &push);
            vkCmdDraw(cmd, CUBE_VERTS, 1, 0, 0);
        }

        vkCmdEndRendering(cmd);

        std::vector<OverlayVertex> overlayVerts;
        overlayVerts.reserve(2048);

        // Graph area: bottom 25% of screen.
        // Vulkan NDC: y=-1 top, y=+1 bottom.
        static constexpr float GRAPH_TOP    =  0.5f;  // 75% down from top
        static constexpr float GRAPH_BOTTOM =  1.0f;  // screen bottom edge
        static constexpr float GRAPH_LEFT   = -1.0f;
        static constexpr float GRAPH_RIGHT  =  1.0f;
        static constexpr float GRAPH_HEIGHT = GRAPH_BOTTOM - GRAPH_TOP; // +0.5

        // Opaque background quad (blocks cubes from bleeding through).
        pushQuad(overlayVerts, GRAPH_LEFT, GRAPH_TOP, GRAPH_RIGHT, GRAPH_BOTTOM,
                 0.08f, 0.08f, 0.10f, 1.0f);

        // Auto-scale: find peak among recent samples, clamp to minimum.
        float graphMaxMs = GRAPH_MIN_SCALE_MS;
        for (float t : graphTimes) {
            if (t > graphMaxMs) graphMaxMs = t;
        }
        graphMaxMs *= 1.2f; // 20% headroom above peak

        // 16ms reference line (60fps target). Grows upward from bottom.
        float refY = GRAPH_BOTTOM - (16.0f / graphMaxMs) * GRAPH_HEIGHT;
        if (refY > GRAPH_TOP) { // only draw if it fits within the graph
            pushQuad(overlayVerts, GRAPH_LEFT, refY, GRAPH_RIGHT, refY + 0.003f,
                     0.3f, 0.3f, 0.3f);
        }

        // Frame time bars (grow upward from bottom edge).
        float barWidth = (GRAPH_RIGHT - GRAPH_LEFT) / static_cast<float>(GRAPH_SAMPLES);
        for (int i = 0; i < GRAPH_SAMPLES; ++i) {
            int sampleIdx = (graphIndex + i) % GRAPH_SAMPLES;
            float ms = graphTimes[static_cast<std::size_t>(sampleIdx)];
            if (ms <= 0.0f) continue;

            float barFrac = ms / graphMaxMs;
            if (barFrac > 1.0f) barFrac = 1.0f;

            float x0 = GRAPH_LEFT + static_cast<float>(i) * barWidth;
            float x1 = x0 + barWidth * 0.8f;
            float y0 = GRAPH_BOTTOM;                       // bottom edge
            float y1 = GRAPH_BOTTOM - barFrac * GRAPH_HEIGHT; // grows up

            // Green < 16ms, yellow 16-33ms, red > 33ms.
            float r, g, b;
            if (ms < 16.0f)       { r = 0.2f; g = 0.9f; b = 0.2f; }
            else if (ms < 33.0f)  { r = 0.9f; g = 0.9f; b = 0.2f; }
            else                  { r = 0.9f; g = 0.2f; b = 0.2f; }

            pushQuad(overlayVerts, x0, y1, x1, y0, r, g, b);
        }

        // Status dots (mode 5 only): just above the graph.
        if (mode == Mode::AutoWarmStats) {
            static constexpr float DOT_TOP    = 0.44f;
            static constexpr float DOT_BOTTOM = 0.48f;
            float dotWidth = (GRAPH_RIGHT - GRAPH_LEFT - 0.1f)
                           / static_cast<float>(TOTAL_PERMS);
            float dotStart = GRAPH_LEFT + 0.05f;

            for (int p = 0; p < TOTAL_PERMS; ++p) {
                float dx0 = dotStart + static_cast<float>(p) * dotWidth;
                float dx1 = dx0 + dotWidth * 0.7f;

                float r, g, b;
                if (!spawned[static_cast<std::size_t>(p)]) {
                    // Not yet spawned: dark gray.
                    r = 0.25f; g = 0.25f; b = 0.25f;
                } else {
                    auto it = asyncPipelines.find(p);
                    if (it != asyncPipelines.end() && it->second.isOptimized()) {
                        // Fully optimized: green.
                        r = 0.2f; g = 0.9f; b = 0.2f;
                    } else {
                        // Fast-linked, optimization pending: yellow.
                        r = 0.9f; g = 0.9f; b = 0.2f;
                    }
                }
                pushQuad(overlayVerts, dx0, DOT_TOP, dx1, DOT_BOTTOM, r, g, b);
            }
        }

        // Upload overlay vertices.
        std::uint32_t overlayVertCount = static_cast<std::uint32_t>(overlayVerts.size());
        if (overlayVertCount > MAX_OVERLAY_VERTS)
            overlayVertCount = static_cast<std::uint32_t>(MAX_OVERLAY_VERTS);
        if (overlayVertCount > 0) {
            auto& overlayVB = overlayVBs[frame.index];
            std::memcpy(overlayVB.mappedData(), overlayVerts.data(),
                        overlayVertCount * sizeof(OverlayVertex));
        }

        // Render overlay.
        VkRenderingAttachmentInfo overlayColor{};
        overlayColor.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        overlayColor.imageView   = img.view;
        overlayColor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        overlayColor.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
        overlayColor.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo overlayRI{};
        overlayRI.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        overlayRI.renderArea           = {{0, 0}, swapchain.extent()};
        overlayRI.layerCount           = 1;
        overlayRI.colorAttachmentCount = 1;
        overlayRI.pColorAttachments    = &overlayColor;

        vkCmdBeginRendering(cmd, &overlayRI);

        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        if (overlayVertCount > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              overlayPipeline.vkPipeline());
            VkBuffer ovb = overlayVBs[frame.index].vkBuffer();
            VkDeviceSize ovbOffset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &ovb, &ovbOffset);
            vkCmdDraw(cmd, overlayVertCount, 1, 0, 0);
        }

        vkCmdEndRendering(cmd);

        vksdl::transitionToPresent(cmd, img.image);
        vksdl::endCommands(cmd);

        vksdl::presentFrame(device, swapchain, window, frame, img,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        // Measure render time (acquire through present, excludes event polling).
        auto renderEnd = Clock::now();
        lastDtMs = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
        graphTimes[static_cast<std::size_t>(graphIndex)] = lastDtMs;
        graphIndex = (graphIndex + 1) % GRAPH_SAMPLES;

        // Update window title (throttled to ~1/sec to avoid DWM title redraw stalls).
        ++frameNum;
        {
            float sinceTitleUpdate = std::chrono::duration<float>(
                renderEnd - lastTitleUpdate).count();
            if (sinceTitleUpdate >= 1.0f) {
                lastTitleUpdate = renderEnd;

                float sum = 0;
                int count = 0;
                for (float t : graphTimes) {
                    if (t > 0) { sum += t; ++count; }
                }
                float avg = count > 0 ? sum / static_cast<float>(count) : 0.0f;
                float fps = avg > 0.0f ? 1000.0f / avg : 0.0f;

                int optimized = 0;
                for (auto& [k, h] : asyncPipelines) {
                    if (h.isOptimized()) ++optimized;
                }

                bool cold = (mode == Mode::MonoCold || mode == Mode::AutoCold);
                auto& cacheRef = cold ? coldCache : warmCache;

                char buf[512];
                if (isMono(mode) || mode == Mode::NoSurprises) {
                    int misses = mode == Mode::NoSurprises ? noSurprisesMisses : 0;
                    if (mode == Mode::NoSurprises) {
                        std::snprintf(buf, sizeof(buf),
                            "Pipeline Compiler | %s | %d/%d (%d misses) | %.0f FPS (%.1fms) | Cache: %zu bytes",
                            modeName(mode), spawnNext, TOTAL_PERMS, misses, fps, avg,
                            cacheRef.dataSize());
                    } else {
                        std::snprintf(buf, sizeof(buf),
                            "Pipeline Compiler | %s | %d/%d spawned | %.0f FPS (%.1fms) | Cache: %zu bytes",
                            modeName(mode), spawnNext, TOTAL_PERMS, fps, avg,
                            cacheRef.dataSize());
                    }
                } else {
                    std::snprintf(buf, sizeof(buf),
                        "Pipeline Compiler | %s | %d/%d spawned (%d optimized) | %.0f FPS (%.1fms) | Cache: %zu bytes",
                        modeName(mode), spawnNext, TOTAL_PERMS, optimized, fps, avg,
                        cacheRef.dataSize());
                }
                SDL_SetWindowTitle(window.sdlWindow(), buf);
            }
        }
    }

    // Print final stats if not yet printed (quit before all spawned).
    if (!statsPrinted && !spawnRecords.empty()) {
        printStats();
    }

    device.waitIdle();

    // Persist warm cache for next run.
    // Merge cold cache data (if any) into warmCache so cold-mode compilations
    // are available on the next warm run. This is the key to --cold / --warm workflow.
    compiler.reset();
    {
        VkPipelineCache src = coldCache.vkPipelineCache();
        vkMergePipelineCaches(device.vkDevice(),
                             warmCache.vkPipelineCache(), 1, &src);
    }
    {
        using SysClock = std::chrono::high_resolution_clock;
        auto saveStart = SysClock::now();
        auto saveResult = warmCache.save(cachePath);
        auto saveEnd = SysClock::now();
        double saveMs = std::chrono::duration<double, std::milli>(saveEnd - saveStart).count();
        if (saveResult.ok()) {
            std::printf("Warm cache saved (%zu bytes, %.1fms)\n",
                        warmCache.dataSize(), saveMs);
        } else {
            std::fprintf(stderr, "Failed to save cache: %s\n",
                         saveResult.error().message.c_str());
        }
    }

    return 0;
}
