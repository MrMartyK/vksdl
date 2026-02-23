add_executable(descriptor_pool main.cpp)
target_link_libraries(descriptor_pool PRIVATE vksdl glm::glm)

# --- Compile GLSL shaders to SPIR-V ---

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})

set(SHADERS
    ${SHADER_DIR}/pool_cube.vert
    ${SHADER_DIR}/pool_cube.frag
)

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV ${SHADER_OUT}/${SHADER_NAME}.spv)
    set(SPIRV_VALIDATE_CMD "")
    if(SPIRV_VAL)
        set(SPIRV_VALIDATE_CMD COMMAND ${SPIRV_VAL} ${SPV})
    endif()
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV}
        ${SPIRV_VALIDATE_CMD}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} -> ${SHADER_NAME}.spv"
    )
    list(APPEND SPV_FILES ${SPV})
endforeach()

add_custom_target(descriptor_pool_shaders ALL DEPENDS ${SPV_FILES})
add_dependencies(descriptor_pool descriptor_pool_shaders)

add_custom_command(TARGET descriptor_pool POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_OUT} $<TARGET_FILE_DIR:descriptor_pool>/shaders
)
