cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch-example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_executable(swapper ./pageswapper/attention-swapper.c)

add_library(torch-lib-custom STATIC ./torch-lib-custom/torch-example.cpp)
target_include_directories(torch-lib-custom PRIVATE ./torch-lib-custom)

target_link_libraries(swapper torch-lib-custom)
target_link_libraries(swapper "${TORCH_LIBRARIES}")

set_property(TARGET torch-lib-custom PROPERTY CXX_STANDARD 17)
