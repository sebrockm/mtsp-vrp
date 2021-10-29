cmake_minimum_required(VERSION 3.21)

project(mtsp-vpr)

set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(conan-trial conan-trial.cpp)
target_link_libraries(conan-trial PRIVATE ${CONAN_LIBS})

add_subdirectory(tsplp)

message(STATUS ${CONAN_LIBS})
