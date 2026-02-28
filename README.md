<div align="center">

# vksdl

**Vulkan without the 800 lines of boilerplate before your first triangle.**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![Vulkan 1.3](https://img.shields.io/badge/Vulkan-1.3-red?logo=vulkan&logoColor=white)](https://vulkan.lunarg.com/)
[![SDL3](https://img.shields.io/badge/SDL-3-blue?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48L3N2Zz4=)](https://www.libsdl.org/)
[![License: Zlib](https://img.shields.io/badge/License-Zlib-brightgreen.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.12.0-orange)](https://github.com/MrMartyK/vksdl/releases)
[![Tests](https://img.shields.io/badge/tests-43%20passing-success)]()

<br>

vksdl wraps the ceremony -- instance creation, device selection, swapchain management,
synchronization, pipeline construction -- and leaves the actual rendering to you.

One `#include`, raw `VkCommandBuffer` inside, full escape hatches everywhere.

[Website](https://vksdl.com) · [Examples](#examples) · [Documentation](#api-at-a-glance) · [Getting Started](#getting-started)

</div>

---

<br>

## Why vksdl

Vulkan is explicit by design. That's its strength. But a huge amount of Vulkan code has exactly one correct answer -- creating an instance, selecting a GPU, tearing down objects in the right order. vksdl wraps that ceremony and leaves the real decisions to you.

| | |
|---|---|
| **~17k lines** of C++20 across 56 public headers | **43 tests**, 20 working examples |
| **Vulkan 1.3 core** -- dynamic rendering, synchronization2, timeline semaphores | **Zero per-frame allocations** in the hot path |
| **SDL3** windowing -- Windows, Linux, macOS from one codebase | No legacy render passes, no compatibility mode |

<br>

## Getting Started

### Prerequisites

- [Vulkan SDK](https://vulkan.lunarg.com/) 1.3+
- CMake 3.21+
- C++20 compiler (GCC 14+, Clang 18+, MSVC 2022+)

### Build

```bash
git clone --recursive https://github.com/MrMartyK/vksdl.git
cd vksdl
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
cd build && ctest --output-on-failure
```

Dependencies (SDL3, VMA, stb, cgltf, tinyobjloader) are fetched automatically via the bundled vcpkg submodule.

<br>

## Setup Is 15 Lines. The Rest Is Your Vulkan.

```cpp
auto app       = vksdl::App::create().value();
auto window    = app.createWindow("Triangle", 1280, 720).value();

auto instance  = vksdl::InstanceBuilder{}.appName("tri")
    .requireVulkan(1, 3).enableWindowSupport().build().value();

auto surface   = vksdl::Surface::create(instance, window).value();

auto device    = vksdl::DeviceBuilder(instance, surface)
    .needSwapchain().needDynamicRendering().needSync2()
    .preferDiscreteGpu().build().value();

auto swapchain = vksdl::SwapchainBuilder(device, surface)
    .size(window.pixelSize()).build().value();

auto frames    = vksdl::FrameSync::create(device, swapchain.imageCount()).value();

auto pipeline  = vksdl::PipelineBuilder(device)
    .vertexShader("shaders/triangle.vert.spv")
    .fragmentShader("shaders/triangle.frag.spv")
    .colorFormat(swapchain).build().value();
```

Then your render loop is standard Vulkan:

```cpp
auto [frame, img] = vksdl::acquireFrame(swapchain, frames, device, window).value();
vksdl::beginOneTimeCommands(frame.cmd);

// --- This is real Vulkan. You own the command buffer. ---
vksdl::transitionToColorAttachment(frame.cmd, img.image);
// vkCmdBeginRendering, vkCmdDraw, vkCmdEndRendering -- your code, your decisions
vksdl::transitionToPresent(frame.cmd, img.image);

vksdl::endCommands(frame.cmd);
vksdl::presentFrame(device, swapchain, window, frame, img,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
```

The full triangle example is [113 lines](examples/triangle/main.cpp) including the render loop and resize handling. The raw Vulkan equivalent is 800+.

<br>

## What Gets Wrapped vs What Stays Raw

| vksdl handles this | You write this |
|---|---|
| Instance + validation + debug messenger | Nothing -- one builder call |
| GPU selection, queue families, feature chains | `needSwapchain()`, `needRayTracingPipeline()` |
| Swapchain format/present mode, image views, resize | `recreate()` on window resize |
| Fences, semaphores, round-robin acquire | Nothing -- `acquireFrame()` / `presentFrame()` |
| SPIR-V loading, pipeline layout, blend/cull defaults | Record commands, bind, draw |
| VMA allocation, typed buffer/image builders | Choose usage, upload data |
| BLAS/TLAS construction, SBT layout, RT pipeline | Trace rays, write shaders |
| Render graph: barriers, resource lifetime, toposort | Declare passes, record in callbacks |

Every RAII object exposes its raw Vulkan handle -- `vkDevice()`, `vkPipeline()`, `vkBuffer()`, `vkImage()` -- so you can drop to raw Vulkan anywhere vksdl doesn't cover your case.

<br>

## Examples

| Example | What it demonstrates | Lines |
|---|---|---:|
| [triangle](examples/triangle/) | Window, device, swapchain, pipeline, render loop | 113 |
| [quad](examples/quad/) | Vertex/index buffers, VMA staged uploads | ~145 |
| [compute](examples/compute/) | Compute shader, storage image, blit to swapchain | 129 |
| [cube](examples/cube/) | 3D depth, uniform buffers, descriptor sets | ~200 |
| [textured_cube](examples/textured_cube/) | Texture loading, mipmaps, samplers | ~250 |
| [multi_object](examples/multi_object/) | Dynamic UBOs, multiple descriptor sets, debug names | ~300 |
| [msaa](examples/msaa/) | MSAA with inline resolve, dynamic recreation | ~180 |
| [model](examples/model/) | glTF/OBJ loading, PBR materials, directional lighting | ~220 |
| [rt_triangle](examples/rt_triangle/) | Minimal ray tracing: BLAS, TLAS, SBT, traceRays | ~300 |
| [rt_spheres](examples/rt_spheres/) | Path tracer: 475 spheres, GGX, physical sky, DOF | 593 |
| [deferred](examples/deferred/) | 40-pass render graph: shadow, G-buffer, lighting, tonemap | 709 |
| [pipeline_compiler](examples/pipeline_compiler/) | GPL fast-linking, async compile, pipeline feedback | 1339 |

<details>
<summary><strong>8 more examples</strong></summary>

<br>

Pipeline cache, timeline sync, dynamic state, descriptor pools, async transfer, unified layouts, shader reflection, and device fault diagnostics.

</details>

<br>

## API at a Glance

<details>
<summary><strong>Core Types (60+)</strong></summary>

<br>

All types are RAII, move-only, and return `Result<T>` by default.

`orThrow()` is an optional escape hatch -- it throws when exceptions are enabled and fail-fasts when exceptions are disabled.

**Initialization** -- `App`, `InstanceBuilder`, `Surface`, `DeviceBuilder`

**Presentation** -- `SwapchainBuilder`, `FrameSync`, `acquireFrame`, `presentFrame`

**Pipelines** -- `PipelineBuilder`, `ComputePipelineBuilder`, `RTPipelineBuilder`, `PipelineCache`

**Resources** -- `Buffer`, `Image`, `Sampler`, `DescriptorSetLayout`, `DescriptorPool`

**Ray Tracing** -- `Blas`, `Tlas`, `ShaderBindingTable`

**Render Graph** -- `RenderGraph`, `RenderPass`, automatic barrier insertion, topological sort

**Utilities** -- `ShaderModule`, `TimelineSemaphore`, `QueryPool`, `DebugName`

</details>

<br>

## Design Philosophy

**Wrap ceremony. Leave intent raw.**

*Ceremony* is code with one correct answer: creating an instance, selecting a GPU, destroying objects in the right order. vksdl wraps that.

*Intent* is code where you're making real choices: recording commands, choosing wait stages, structuring submissions. vksdl leaves that alone.

The test: if two experienced Vulkan developers would write the same boilerplate identically, vksdl should eliminate it. If they'd write it differently, vksdl stays out of the way.

<br>

## Platform Support

| Platform | Compiler | Status |
|---|---|---|
| Windows 11 | GCC 15.2 (MSYS2 MinGW) | Tested, RTX 3060 |
| Linux | GCC 14 / Clang 18 | Tested via CI |
| macOS | Clang (via SDL3) | Expected to work, not yet tested |

<br>

## Project Status

**v0.12.0** -- Core API is stable across 60+ wrapped types. The render graph and pipeline model are functional and tested but still evolving.

See the [changelog](CHANGELOG.md) for release history.

<br>

## License

Released under the [Zlib License](LICENSE).

<br>

---

<div align="center">

[vksdl.com](https://vksdl.com)

</div>
