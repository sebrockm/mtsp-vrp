cmake_minimum_required(VERSION 3.21)

project(mtsp-vrp)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

if (MSVC)
    set(warning_flags
        /permissive- /W4 /w14242 /w14254 /w14263 /w14265 /w14287 /we4289 /w14296 /w14311 /w14545 /w14546 /w14547 /w14549 /w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928)
    set(warnings_as_erros_flag /WX)
    set(no_warnings_as_erros_flag /WX-)
else()
    set(warning_flags
        -pedantic -Wall -Wextra
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wmisleading-indentation
        -Wnull-dereference -Wdouble-promotion -Wformat=2)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(warning_flags ${warning_flags} -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast)
    endif()
    set(warnings_as_erros_flag -Werror)
    set(no_warnings_as_erros_flag -Wno-error)
endif()

add_compile_options(${warning_flags} ${warnings_as_erros_flag})

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# disable warnings as errors for xtensor as it yields too many warnings
# apparently this is only an issue on MSVC
if (MSVC)
    target_compile_options(CONAN_PKG::xtensor INTERFACE ${no_warnings_as_erros_flag})
endif()

add_subdirectory(mtsp-vrp-c)
add_subdirectory(tsplib)
add_subdirectory(tsplp)
