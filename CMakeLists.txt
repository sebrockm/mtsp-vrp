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

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# workaround to get access to a boost/graph fix before it is release
if(WIN32)
    execute_process(
        COMMAND cmd /c bootstrap.bat && b2 headers
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/boost
        ECHO_OUTPUT_VARIABLE
        ECHO_ERROR_VARIABLE
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(boost_dir ${CMAKE_SOURCE_DIR}/boost)
else()
    execute_process(
        COMMAND ./bootstrap.sh --prefix=${CMAKE_BINARY_DIR}/boost && ./b2 headers --prefix=${CMAKE_BINARY_DIR}/boost
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/boost
        ECHO_OUTPUT_VARIABLE
        ECHO_ERROR_VARIABLE
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(boost_dir ${CMAKE_BINARY_DIR}/boost/include)
    execute_process(COMMAND ls ${boost_dir})
    execute_process(COMMAND ls ${boost_dir}/boost)
endif()
add_library(my-boost INTERFACE)
target_include_directories(my-boost INTERFACE ${boost_dir})

add_subdirectory(mtsp-vrp-c)
add_subdirectory(python)
add_subdirectory(tsplp)
