add_executable(msaa main.cpp)
target_link_libraries(msaa PRIVATE vksdl)

# Reuse triangle shaders -- MSAA is pipeline state, not shader state.
add_dependencies(msaa triangle_shaders)
add_custom_command(TARGET msaa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:msaa>/shaders
)
