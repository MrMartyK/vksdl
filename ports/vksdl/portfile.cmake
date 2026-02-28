vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MrMartyK/vksdl
    REF 79c7e5c46f2a9c04daeebb8812edfb908886e3f5
    SHA512 2d0937dac75d4a251406de6dff7403fea6debe7c361a55bbd7b4e37cf57338eeda366daf0a4512e1a081bb3d3bbc11e5d00f3b443563fa939e88225958a09233
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DVKSDL_BUILD_TESTS=OFF
        -DVKSDL_BUILD_EXAMPLES=OFF
        -DVKSDL_WERROR=OFF
        -DVKSDL_INSTALL=ON
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME vksdl CONFIG_PATH lib/cmake/vksdl)

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
