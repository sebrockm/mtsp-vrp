cmake_minimum_required(VERSION 3.21)

file(GLOB tests *.cpp)

add_executable(graph-algos-test ${tests})

target_link_libraries(graph-algos-test PRIVATE graph-algos CONAN_PKG::catch2)

target_precompile_headers(graph-algos-test
	PRIVATE
	<boost/graph/connected_components.hpp>
	<boost/graph/undirected_dfs.hpp>
	<catch2/catch.hpp>

	<algorithm>
	<array>
	<span>
	<unordered_map>
	<utility>
)

set_source_files_properties(catch-main.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

add_test(NAME graph-algos-test COMMAND graph-algos-test)
