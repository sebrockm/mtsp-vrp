cmake_minimum_required(VERSION 3.25)

project(mtsp-vrp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

if (MSVC)
    set(warning_flags
        /permissive- /W4 /w14242 /w14254 /w14263 /w14265 /w14287 /we4289 /w14296 /w14311 /w14545 /w14546 /w14547 /w14549 /w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928)

    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.14) # MSVC supports SYSTEM include dirs
        set(warning_flags ${warning_flags} /wd4701 /wd4267 /wd4244) # these warnings are still reported from within SYSTEM include dirs, so disable them)
        set(warnings_as_erros_flag /WX)
        if (CMAKE_VERSION VERSION_LESS 3.24) # Only from 3.24 onwards, MSVC's SYSTEM includes are supported in cmake
            # for older cmake versions, use this workaround mentioned in https://gitlab.kitware.com/cmake/cmake/-/issues/17904
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /experimental:external /external:W0 ")
            set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I ")
        endif()
    else() # MSVS doesn't support SYSTEM include dirs, so we cannot treat warnings as errors
        set(warnings_as_erros_flag "")
    endif()
else()
    set(warning_flags
        -pedantic -Wall -Wextra
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wmisleading-indentation
        -Wnull-dereference -Wdouble-promotion -Wformat=2)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(warning_flags ${warning_flags} -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast)
    endif()
    set(warnings_as_erros_flag -Werror)
endif()

add_compile_options(${warning_flags} ${warnings_as_erros_flag})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
        "${CMAKE_BINARY_DIR}/conan.cmake"
        TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)
conan_cmake_autodetect(settings)
conan_cmake_install(PATH_OR_REFERENCE ..
    BUILD missing
    REMOTE conancenter
)

find_package(Boost 1.80.0)
find_package(Catch2 2.13.7)
find_package(coin-clp 1.17.6)
find_package(xtensor 0.23.10)

# Code Coverage Configuration
add_library(coverage_config INTERFACE)

option(CODE_COVERAGE "Enable coverage reporting" OFF)

if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(coverage_config INTERFACE --coverage)
endif()

set(CMAKE_CXX_CLANG_TIDY clang-tidy -fix -format-style=file -checks=-*,modernize*,-modernize-use-trailing-return-type -p ${CMAKE_BINARY_DIR} -header-filter=.* -warnings-as-errors=*)

# boost*
# bugprone*
# clang-analyzer*,
# cppcore*,
# google*,
# hicpp*,
# llvm*,
# misc*,
# performance*,
# portability*,
# readability*,


add_subdirectory(graph-algos)
add_subdirectory(mtsp-vrp-c)
add_subdirectory(python)
add_subdirectory(tsplp)
