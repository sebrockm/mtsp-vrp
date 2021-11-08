cmake_minimum_required(VERSION 3.21)

project(mtsp-vrp)

enable_testing()

set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_executable(conan-trial conan-trial.cpp)
target_link_libraries(conan-trial PRIVATE ${CONAN_TARGETS} tsplp)

add_subdirectory(tsplp)
