add_executable(test_error unit/test_error.cpp)
target_link_libraries(test_error PRIVATE vksdl)
add_test(NAME test_error COMMAND test_error)

add_executable(test_result unit/test_result.cpp)
target_link_libraries(test_result PRIVATE vksdl)
add_test(NAME test_result COMMAND test_result)

add_executable(test_native_accessors unit/test_native_accessors.cpp)
target_link_libraries(test_native_accessors PRIVATE vksdl)
add_test(NAME test_native_accessors COMMAND test_native_accessors)

add_executable(test_projection unit/test_projection.cpp)
target_link_libraries(test_projection PRIVATE vksdl)
add_test(NAME test_projection COMMAND test_projection)

add_executable(test_transform unit/test_transform.cpp)
target_link_libraries(test_transform PRIVATE vksdl)
add_test(NAME test_transform COMMAND test_transform)

add_executable(test_window integration/test_window.cpp)
target_link_libraries(test_window PRIVATE vksdl)
add_test(NAME test_window COMMAND test_window)

add_executable(test_wsi_extensions unit/test_wsi_extensions.cpp)
target_link_libraries(test_wsi_extensions PRIVATE vksdl)
add_test(NAME test_wsi_extensions COMMAND test_wsi_extensions)

add_executable(test_surface integration/test_surface.cpp)
target_link_libraries(test_surface PRIVATE vksdl)
add_test(NAME test_surface COMMAND test_surface)

add_executable(test_instance_builder integration/test_instance_builder.cpp)
target_link_libraries(test_instance_builder PRIVATE vksdl)
add_test(NAME test_instance_builder COMMAND test_instance_builder)

add_executable(test_device_builder integration/test_device_builder.cpp)
target_link_libraries(test_device_builder PRIVATE vksdl)
add_test(NAME test_device_builder COMMAND test_device_builder)

add_executable(test_swapchain integration/test_swapchain.cpp)
target_link_libraries(test_swapchain PRIVATE vksdl)
add_test(NAME test_swapchain COMMAND test_swapchain)

add_executable(test_framesync integration/test_framesync.cpp)
target_link_libraries(test_framesync PRIVATE vksdl)
add_test(NAME test_framesync COMMAND test_framesync)

add_executable(test_allocator integration/test_allocator.cpp)
target_link_libraries(test_allocator PRIVATE vksdl)
add_test(NAME test_allocator COMMAND test_allocator)

# --- Pipeline test (needs compiled shaders) ---

add_executable(test_pipeline integration/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE vksdl)
add_test(NAME test_pipeline COMMAND test_pipeline)

# Reuse triangle shaders -- copy compiled .spv files next to test executable.
# The triangle example already compiles them; we just depend on that target
# and copy the output.
add_dependencies(test_pipeline triangle_shaders)
add_custom_command(TARGET test_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_pipeline>/shaders
)

# --- Device fault test ---

add_executable(test_device_fault integration/test_device_fault.cpp)
target_link_libraries(test_device_fault PRIVATE vksdl)
add_test(NAME test_device_fault COMMAND test_device_fault)

# --- Pipeline cache test (reuses triangle shaders) ---

add_executable(test_pipeline_cache integration/test_pipeline_cache.cpp)
target_link_libraries(test_pipeline_cache PRIVATE vksdl)
add_test(NAME test_pipeline_cache COMMAND test_pipeline_cache)

add_dependencies(test_pipeline_cache triangle_shaders)
add_custom_command(TARGET test_pipeline_cache POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_pipeline_cache>/shaders
)

# --- Compute pipeline test (needs compiled compute shader) ---

add_executable(test_compute_pipeline integration/test_compute_pipeline.cpp)
target_link_libraries(test_compute_pipeline PRIVATE vksdl)
add_test(NAME test_compute_pipeline COMMAND test_compute_pipeline)

set(COMP_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/integration/shaders)
set(COMP_SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${COMP_SHADER_OUT})

set(SPIRV_VALIDATE_NOOP_CMD "")
if(SPIRV_VAL)
    set(SPIRV_VALIDATE_NOOP_CMD COMMAND ${SPIRV_VAL} ${COMP_SHADER_OUT}/noop.comp.spv)
endif()
add_custom_command(
    OUTPUT ${COMP_SHADER_OUT}/noop.comp.spv
    COMMAND ${GLSLC} ${COMP_SHADER_DIR}/noop.comp -o ${COMP_SHADER_OUT}/noop.comp.spv
    ${SPIRV_VALIDATE_NOOP_CMD}
    DEPENDS ${COMP_SHADER_DIR}/noop.comp
    COMMENT "Compiling noop.comp -> noop.comp.spv"
)
set(SPIRV_VALIDATE_REFLECT_TEST_CMD "")
if(SPIRV_VAL)
    set(SPIRV_VALIDATE_REFLECT_TEST_CMD COMMAND ${SPIRV_VAL} ${COMP_SHADER_OUT}/reflect_test.comp.spv)
endif()
add_custom_command(
    OUTPUT ${COMP_SHADER_OUT}/reflect_test.comp.spv
    COMMAND ${GLSLC} ${COMP_SHADER_DIR}/reflect_test.comp -o ${COMP_SHADER_OUT}/reflect_test.comp.spv
    ${SPIRV_VALIDATE_REFLECT_TEST_CMD}
    DEPENDS ${COMP_SHADER_DIR}/reflect_test.comp
    COMMENT "Compiling reflect_test.comp -> reflect_test.comp.spv"
)
add_custom_target(test_compute_shaders ALL
    DEPENDS
        ${COMP_SHADER_OUT}/noop.comp.spv
        ${COMP_SHADER_OUT}/reflect_test.comp.spv
)
add_dependencies(test_compute_pipeline test_compute_shaders)

add_custom_command(TARGET test_compute_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${COMP_SHADER_OUT}
        $<TARGET_FILE_DIR:test_compute_pipeline>/shaders
)

# --- Descriptor set test (reuses triangle shaders for pipeline integration) ---

add_executable(test_descriptor_set integration/test_descriptor_set.cpp)
target_link_libraries(test_descriptor_set PRIVATE vksdl)
add_test(NAME test_descriptor_set COMMAND test_descriptor_set)

add_dependencies(test_descriptor_set triangle_shaders)
add_custom_command(TARGET test_descriptor_set POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_descriptor_set>/shaders
)

# --- Sampler test ---

add_executable(test_sampler integration/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE vksdl)
add_test(NAME test_sampler COMMAND test_sampler)

# --- Texture upload test (needs test assets) ---

add_executable(test_texture_upload integration/test_texture_upload.cpp)
target_link_libraries(test_texture_upload PRIVATE vksdl)
add_test(NAME test_texture_upload COMMAND test_texture_upload)

add_custom_command(TARGET test_texture_upload POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/assets
        $<TARGET_FILE_DIR:test_texture_upload>/assets
)

# --- Mipmap test (needs test assets) ---

add_executable(test_mipmaps integration/test_mipmaps.cpp)
target_link_libraries(test_mipmaps PRIVATE vksdl)
add_test(NAME test_mipmaps COMMAND test_mipmaps)

add_custom_command(TARGET test_mipmaps POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/assets
        $<TARGET_FILE_DIR:test_mipmaps>/assets
)

# --- Dynamic UBO test ---

add_executable(test_dynamic_ubo integration/test_dynamic_ubo.cpp)
target_link_libraries(test_dynamic_ubo PRIVATE vksdl)
add_test(NAME test_dynamic_ubo COMMAND test_dynamic_ubo)

# --- Debug name test ---

add_executable(test_debug_name integration/test_debug_name.cpp)
target_link_libraries(test_debug_name PRIVATE vksdl)
add_test(NAME test_debug_name COMMAND test_debug_name)

# --- MSAA test (reuses triangle shaders for pipeline creation) ---

add_executable(test_msaa integration/test_msaa.cpp)
target_link_libraries(test_msaa PRIVATE vksdl)
add_test(NAME test_msaa COMMAND test_msaa)

add_dependencies(test_msaa triangle_shaders)
add_custom_command(TARGET test_msaa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_msaa>/shaders
)

# --- Query pool test ---

add_executable(test_query_pool integration/test_query_pool.cpp)
target_link_libraries(test_query_pool PRIVATE vksdl)
add_test(NAME test_query_pool COMMAND test_query_pool)

# --- Mesh loading test (needs Box.glb asset) ---

add_executable(test_mesh integration/test_mesh.cpp)
target_link_libraries(test_mesh PRIVATE vksdl)
add_test(NAME test_mesh COMMAND test_mesh)

add_custom_command(TARGET test_mesh POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/assets
        $<TARGET_FILE_DIR:test_mesh>/assets
)

# --- Dynamic state test (reuses triangle shaders) ---

add_executable(test_dynamic_state integration/test_dynamic_state.cpp)
target_link_libraries(test_dynamic_state PRIVATE vksdl)
add_test(NAME test_dynamic_state COMMAND test_dynamic_state)

add_dependencies(test_dynamic_state triangle_shaders)
add_custom_command(TARGET test_dynamic_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_dynamic_state>/shaders
)

# --- Unified image layouts test ---

add_executable(test_unified_layouts integration/test_unified_layouts.cpp)
target_link_libraries(test_unified_layouts PRIVATE vksdl)
add_test(NAME test_unified_layouts COMMAND test_unified_layouts)

# --- Transfer queue test ---

add_executable(test_transfer_queue integration/test_transfer_queue.cpp)
target_link_libraries(test_transfer_queue PRIVATE vksdl)
add_test(NAME test_transfer_queue COMMAND test_transfer_queue)

# --- Async compute test ---

add_executable(test_async_compute integration/test_async_compute.cpp)
target_link_libraries(test_async_compute PRIVATE vksdl)
add_test(NAME test_async_compute COMMAND test_async_compute)

# --- Descriptor allocator test ---

add_executable(test_descriptor_allocator integration/test_descriptor_allocator.cpp)
target_link_libraries(test_descriptor_allocator PRIVATE vksdl)
add_test(NAME test_descriptor_allocator COMMAND test_descriptor_allocator)

# --- Sampler cache test ---

add_executable(test_sampler_cache integration/test_sampler_cache.cpp)
target_link_libraries(test_sampler_cache PRIVATE vksdl)
add_test(NAME test_sampler_cache COMMAND test_sampler_cache)

# --- Frame descriptor allocator test ---

add_executable(test_frame_descriptor_allocator integration/test_frame_descriptor_allocator.cpp)
target_link_libraries(test_frame_descriptor_allocator PRIVATE vksdl)
add_test(NAME test_frame_descriptor_allocator COMMAND test_frame_descriptor_allocator)

# --- Descriptor pool test ---

add_executable(test_descriptor_pool integration/test_descriptor_pool.cpp)
target_link_libraries(test_descriptor_pool PRIVATE vksdl)
add_test(NAME test_descriptor_pool COMMAND test_descriptor_pool)

# --- Timeline sync test (reuses triangle shaders) ---

add_executable(test_timeline_sync integration/test_timeline_sync.cpp)
target_link_libraries(test_timeline_sync PRIVATE vksdl)
add_test(NAME test_timeline_sync COMMAND test_timeline_sync)

add_dependencies(test_timeline_sync triangle_shaders)
add_custom_command(TARGET test_timeline_sync POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_timeline_sync>/shaders
)

# --- Shader reflection test (needs triangle + compute shaders) ---

add_executable(test_shader_reflect integration/test_shader_reflect.cpp)
target_link_libraries(test_shader_reflect PRIVATE vksdl)
add_test(NAME test_shader_reflect COMMAND test_shader_reflect)

# Compile the reflection test compute shader.
set(REFLECT_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/integration/shaders)
set(REFLECT_SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/reflect_shaders)
file(MAKE_DIRECTORY ${REFLECT_SHADER_OUT})

set(SPIRV_VALIDATE_REFLECT_CMD "")
if(SPIRV_VAL)
    set(SPIRV_VALIDATE_REFLECT_CMD COMMAND ${SPIRV_VAL} ${REFLECT_SHADER_OUT}/reflect_test.comp.spv)
endif()
add_custom_command(
    OUTPUT ${REFLECT_SHADER_OUT}/reflect_test.comp.spv
    COMMAND ${GLSLC} ${REFLECT_SHADER_DIR}/reflect_test.comp -o ${REFLECT_SHADER_OUT}/reflect_test.comp.spv
    ${SPIRV_VALIDATE_REFLECT_CMD}
    DEPENDS ${REFLECT_SHADER_DIR}/reflect_test.comp
    COMMENT "Compiling reflect_test.comp -> reflect_test.comp.spv"
)
add_custom_target(test_reflect_shaders ALL DEPENDS ${REFLECT_SHADER_OUT}/reflect_test.comp.spv)
add_dependencies(test_shader_reflect test_reflect_shaders triangle_shaders)

# Copy triangle shaders and reflect test shader next to executable.
add_custom_command(TARGET test_shader_reflect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_shader_reflect>/shaders
)
add_custom_command(TARGET test_shader_reflect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${REFLECT_SHADER_OUT}
        $<TARGET_FILE_DIR:test_shader_reflect>/shaders
)

# --- RT test (needs compiled RT shaders) ---

add_executable(test_rt integration/test_rt.cpp)
target_link_libraries(test_rt PRIVATE vksdl)
add_test(NAME test_rt COMMAND test_rt)

add_dependencies(test_rt rt_triangle_shaders)
add_custom_command(TARGET test_rt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/rt_triangle/shaders
        $<TARGET_FILE_DIR:test_rt>/shaders
)

# --- Render graph: resource state test (no GPU needed) ---

add_executable(test_resource_state integration/test_resource_state.cpp)
target_link_libraries(test_resource_state PRIVATE vksdl)
add_test(NAME test_resource_state COMMAND test_resource_state)

# --- Render graph: barrier compiler test (no GPU needed) ---

add_executable(test_barrier_compiler integration/test_barrier_compiler.cpp)
target_link_libraries(test_barrier_compiler PRIVATE vksdl)
add_test(NAME test_barrier_compiler COMMAND test_barrier_compiler)

# --- Render graph: full integration test (needs GPU) ---

add_executable(test_render_graph integration/test_render_graph.cpp)
target_link_libraries(test_render_graph PRIVATE vksdl)
add_test(NAME test_render_graph COMMAND test_render_graph)

# --- Pipeline feedback test (reuses triangle + compute shaders) ---

add_executable(test_pipeline_feedback integration/test_pipeline_feedback.cpp)
target_link_libraries(test_pipeline_feedback PRIVATE vksdl)
add_test(NAME test_pipeline_feedback COMMAND test_pipeline_feedback)

add_dependencies(test_pipeline_feedback triangle_shaders test_compute_shaders)
add_custom_command(TARGET test_pipeline_feedback POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_pipeline_feedback>/shaders
)
add_custom_command(TARGET test_pipeline_feedback POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${COMP_SHADER_OUT}
        $<TARGET_FILE_DIR:test_pipeline_feedback>/shaders
)

# --- Pipeline handle test (reuses triangle shaders) ---

add_executable(test_pipeline_handle integration/test_pipeline_handle.cpp)
target_link_libraries(test_pipeline_handle PRIVATE vksdl)
add_test(NAME test_pipeline_handle COMMAND test_pipeline_handle)

add_dependencies(test_pipeline_handle triangle_shaders)
add_custom_command(TARGET test_pipeline_handle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_pipeline_handle>/shaders
)

# --- Bindless table test ---

add_executable(test_bindless_table integration/test_bindless_table.cpp)
target_link_libraries(test_bindless_table PRIVATE vksdl)
add_test(NAME test_bindless_table COMMAND test_bindless_table)

# --- Push descriptor writer test ---

add_executable(test_push_descriptor_writer integration/test_push_descriptor_writer.cpp)
target_link_libraries(test_push_descriptor_writer PRIVATE vksdl)
add_test(NAME test_push_descriptor_writer COMMAND test_push_descriptor_writer)

# --- Pipeline compiler test (reuses triangle shaders) ---

add_executable(test_pipeline_compiler integration/test_pipeline_compiler.cpp)
target_link_libraries(test_pipeline_compiler PRIVATE vksdl)
add_test(NAME test_pipeline_compiler COMMAND test_pipeline_compiler)

add_dependencies(test_pipeline_compiler triangle_shaders)
add_custom_command(TARGET test_pipeline_compiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_pipeline_compiler>/shaders
)

# --- Indirect buffer test ---

add_executable(test_indirect_buffer integration/test_indirect_buffer.cpp)
target_link_libraries(test_indirect_buffer PRIVATE vksdl)
add_test(NAME test_indirect_buffer COMMAND test_indirect_buffer)

# --- Memory budget test ---

add_executable(test_memory_budget integration/test_memory_budget.cpp)
target_link_libraries(test_memory_budget PRIVATE vksdl)
add_test(NAME test_memory_budget COMMAND test_memory_budget)

# --- Memory priority test ---

add_executable(test_memory_priority integration/test_memory_priority.cpp)
target_link_libraries(test_memory_priority PRIVATE vksdl)
add_test(NAME test_memory_priority COMMAND test_memory_priority)

# --- Pipeline binary detection test ---

add_executable(test_pipeline_binary integration/test_pipeline_binary.cpp)
target_link_libraries(test_pipeline_binary PRIVATE vksdl)
add_test(NAME test_pipeline_binary COMMAND test_pipeline_binary)

# --- Queue ownership transfer barrier test ---

add_executable(test_queue_ownership integration/test_queue_ownership.cpp)
target_link_libraries(test_queue_ownership PRIVATE vksdl)
add_test(NAME test_queue_ownership COMMAND test_queue_ownership)

# --- Spec constants test (needs compute + triangle shaders) ---

set(SPIRV_VALIDATE_SPEC_CONST_CMD "")
if(SPIRV_VAL)
    set(SPIRV_VALIDATE_SPEC_CONST_CMD COMMAND ${SPIRV_VAL} ${COMP_SHADER_OUT}/spec_const.comp.spv)
endif()
add_custom_command(
    OUTPUT ${COMP_SHADER_OUT}/spec_const.comp.spv
    COMMAND ${GLSLC} ${COMP_SHADER_DIR}/spec_const.comp -o ${COMP_SHADER_OUT}/spec_const.comp.spv
    ${SPIRV_VALIDATE_SPEC_CONST_CMD}
    DEPENDS ${COMP_SHADER_DIR}/spec_const.comp
    COMMENT "Compiling spec_const.comp -> spec_const.comp.spv"
)
add_custom_target(test_spec_const_shaders ALL
    DEPENDS ${COMP_SHADER_OUT}/spec_const.comp.spv
)

add_executable(test_spec_constants integration/test_spec_constants.cpp)
target_link_libraries(test_spec_constants PRIVATE vksdl)
add_test(NAME test_spec_constants COMMAND test_spec_constants)

add_dependencies(test_spec_constants test_spec_const_shaders triangle_shaders)
add_custom_command(TARGET test_spec_constants POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${COMP_SHADER_OUT}
        $<TARGET_FILE_DIR:test_spec_constants>/shaders
)
add_custom_command(TARGET test_spec_constants POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:test_spec_constants>/shaders
)

# --- Mesh pipeline test ---

add_executable(test_mesh_pipeline integration/test_mesh_pipeline.cpp)
target_link_libraries(test_mesh_pipeline PRIVATE vksdl)
add_test(NAME test_mesh_pipeline COMMAND test_mesh_pipeline)

# --- Event test ---

add_executable(test_event integration/test_event.cpp)
target_link_libraries(test_event PRIVATE vksdl)
add_test(NAME test_event COMMAND test_event)

# --- Error recovery policy test ---

add_executable(test_error_recovery integration/test_error_recovery.cpp)
target_link_libraries(test_error_recovery PRIVATE vksdl)
add_test(NAME test_error_recovery COMMAND test_error_recovery)

# --- Command pool test ---

add_executable(test_command_pool integration/test_command_pool.cpp)
target_link_libraries(test_command_pool PRIVATE vksdl)
add_test(NAME test_command_pool COMMAND test_command_pool)

# --- Command pool factory test ---

add_executable(test_command_pool_factory integration/test_command_pool_factory.cpp)
target_link_libraries(test_command_pool_factory PRIVATE vksdl)
add_test(NAME test_command_pool_factory COMMAND test_command_pool_factory)

# --- Present timing detection test ---

add_executable(test_present_timing integration/test_present_timing.cpp)
target_link_libraries(test_present_timing PRIVATE vksdl)
add_test(NAME test_present_timing COMMAND test_present_timing)
