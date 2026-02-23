add_executable(device_fault main.cpp)
target_link_libraries(device_fault PRIVATE vksdl)

# Reuse triangle shaders.
add_dependencies(device_fault triangle_shaders)
add_custom_command(TARGET device_fault POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/examples/triangle/shaders
        $<TARGET_FILE_DIR:device_fault>/shaders
)
