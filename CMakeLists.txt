cmake_minimum_required(VERSION 3.21)

project(mtsp-vpr)

set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(conan-trial conan-trial.cpp)
target_link_libraries(conan-trial PRIVATE ${CONAN_LIBS})

add_library(tsplp OBJECT
	Model.hpp Model.cpp
	Variable.hpp Variable.cpp)

target_link_libraries(tsplp PRIVATE ${CONAN_LIBS_COIN-CLP})

message(STATUS ${CONAN_LIBS})
