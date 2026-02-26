cmake_minimum_required(VERSION 3.21)
project(vksdl_consumer_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED vksdl_DIR)
    message(FATAL_ERROR "vksdl_DIR must point to the installed vksdlConfig.cmake directory.")
endif()

message(STATUS "consumer: vksdl_DIR=${vksdl_DIR}")
find_package(vksdl CONFIG REQUIRED)

get_target_property(VKSDL_IFACE_INCLUDES vksdl::vksdl INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(VKSDL_IFACE_LINK_LIBS vksdl::vksdl INTERFACE_LINK_LIBRARIES)
message(STATUS "consumer: INTERFACE_INCLUDE_DIRECTORIES=${VKSDL_IFACE_INCLUDES}")
message(STATUS "consumer: INTERFACE_LINK_LIBRARIES=${VKSDL_IFACE_LINK_LIBS}")

add_executable(vksdl_consumer_smoke main.cpp)
target_link_libraries(vksdl_consumer_smoke PRIVATE vksdl::vksdl)
