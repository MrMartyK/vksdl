cmake_minimum_required(VERSION 3.21)
project(vksdl VERSION 0.12.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---
find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_image.h")
find_path(CGLTF_INCLUDE_DIRS "cgltf.h")
find_path(TINYOBJLOADER_INCLUDE_DIRS "tiny_obj_loader.h")
find_package(glm CONFIG REQUIRED)

# --- vksdl library ---
add_library(vksdl STATIC
    src/core/error.cpp
    src/vulkan/instance.cpp
    src/vulkan/surface.cpp
    src/platform/sdl3/app_sdl3.cpp
    src/platform/sdl3/window_sdl3.cpp
    src/platform/sdl3/fly_camera_sdl3.cpp
    src/platform/sdl3/orbit_camera_sdl3.cpp
    src/platform/sdl3/util_sdl3.cpp
    src/vulkan/wsi_sdl3.cpp
    src/vulkan/device.cpp
    src/vulkan/swapchain.cpp
    src/vulkan/frames.cpp
    src/vulkan/pipeline.cpp
    src/vulkan/barriers.cpp
    src/vulkan/allocator.cpp
    src/vulkan/buffer.cpp
    src/vulkan/image.cpp
    src/vulkan/compute_pipeline.cpp
    src/vulkan/descriptor_set.cpp
    src/vulkan/sampler.cpp
    src/vulkan/texture.cpp
    src/vulkan/debug.cpp
    src/vulkan/query_pool.cpp
    src/vulkan/mesh.cpp
    src/vulkan/pipeline_cache.cpp
    src/vulkan/loader_gltf.cpp
    src/vulkan/loader_obj.cpp
    src/vulkan/rt_functions.cpp
    src/vulkan/blas.cpp
    src/vulkan/tlas.cpp
    src/vulkan/rt_pipeline.cpp
    src/vulkan/sbt.cpp
    src/vulkan/timeline_sync.cpp
    src/vulkan/descriptor_allocator.cpp
    src/vulkan/descriptor_pool.cpp
    src/vulkan/descriptor_writer.cpp
    src/vulkan/frame_descriptor_allocator.cpp
    src/vulkan/bindless_table.cpp
    src/vulkan/push_descriptor_writer.cpp
    src/vulkan/sampler_cache.cpp
    src/vulkan/transfer_queue.cpp
    src/vulkan/shader_reflect.cpp
    src/graph/resource_state.cpp
    src/graph/barrier_compiler.cpp
    src/graph/pass.cpp
    src/graph/pass_context.cpp
    src/graph/render_graph.cpp
    src/pipeline_model/pipeline_handle.cpp
    src/pipeline_model/pipeline_compiler.cpp
    src/pipeline_model/gpl_library.cpp
)

target_include_directories(vksdl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vksdl PUBLIC Vulkan::Vulkan SDL3::SDL3)
target_link_libraries(vksdl PRIVATE GPUOpen::VulkanMemoryAllocator)
target_include_directories(vksdl PRIVATE ${STB_INCLUDE_DIRS})
target_include_directories(vksdl PRIVATE ${CGLTF_INCLUDE_DIRS})
target_include_directories(vksdl PRIVATE ${TINYOBJLOADER_INCLUDE_DIRS})

# SPIRV-Reflect (compiled as single TU in shader_reflect.cpp).
# Prefer vendored copy; fall back to Vulkan SDK if present.
set(SPIRV_REFLECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spirv-reflect")
if(NOT EXISTS "${SPIRV_REFLECT_DIR}/spirv_reflect.h" AND DEFINED ENV{VULKAN_SDK})
    set(SPIRV_REFLECT_DIR "$ENV{VULKAN_SDK}/Source/SPIRV-Reflect")
endif()
if(NOT EXISTS "${SPIRV_REFLECT_DIR}/spirv_reflect.h")
    message(FATAL_ERROR "SPIRV-Reflect not found at ${SPIRV_REFLECT_DIR}.\n"
        "Either place it in third_party/spirv-reflect/ or set VULKAN_SDK.")
endif()
target_include_directories(vksdl PRIVATE ${SPIRV_REFLECT_DIR} ${SPIRV_REFLECT_DIR}/include)

# --- SPIR-V tools ---
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
find_program(SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/Bin")

# --- Compiler warnings ---
if(MSVC)
    target_compile_options(vksdl PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(vksdl PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# --- Tests and examples (only when building standalone, not as subdirectory) ---
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()
