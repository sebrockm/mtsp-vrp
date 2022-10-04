cmake_minimum_required(VERSION 3.21)

file(GLOB tests *.cpp)

add_executable(graph-algos-test ${tests})

target_link_libraries(graph-algos-test PRIVATE graph-algos CONAN_PKG::catch2)

target_precompile_headers(graph-algos-test
	PRIVATE
	<catch2/catch.hpp>
)

set_source_files_properties(catch-main.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

add_test(NAME graph-algos-test COMMAND graph-algos-test)
