cmake_minimum_required(VERSION 3.21)

project(mtsp-vrp)

enable_testing()

set(CMAKE_CXX_STANDARD 20)

if (MSVC)
    add_compile_options(/WX /permissive- /W4
        /w14242 /w14254 /w14263 /w14265 /w14287 /we4289 /w14296 /w14311 /w14545 /w14546 /w14547 /w14549 /w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928)
else()
    add_compile_options(-Werror -pedantic -Wall -Wextra
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wmisleading-indentation
        -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2)
endif()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# disable warnings as errors for xtensor as it yields too many warnings
# maybe declare the include files as SYSTEM?
if (MSVC)
    target_compile_options(CONAN_PKG::xtensor INTERFACE /WX-)
else()
    target_compile_options(CONAN_PKG::xtensor INTERFACE -Wno-error)
endif()

add_subdirectory(tsplp)
