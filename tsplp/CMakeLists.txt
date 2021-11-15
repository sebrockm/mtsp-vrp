cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.hpp)
file(GLOB sources src/*.cpp src/*.hpp)

add_library(tsplp OBJECT ${headers} ${sources})

target_include_directories(tsplp PUBLIC include)

target_link_libraries(tsplp
	PRIVATE CONAN_PKG::coin-clp CONAN_PKG::boost
	PUBLIC CONAN_PKG::xtensor)

add_subdirectory(test)
