cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.hpp)
file(GLOB sources src/*.cpp src/*.hpp)

add_library(graph-algos STATIC ${headers} ${sources})

target_include_directories(graph-algos PUBLIC include)

target_link_libraries(graph-algos PUBLIC CONAN_PKG::boost)

target_precompile_headers(graph-algos
	PUBLIC
	<boost/graph/adjacency_list.hpp>

	<functional>
	<span>

	PRIVATE
	<boost/graph/adjacency_matrix.hpp>
	<boost/graph/boykov_kolmogorov_max_flow.hpp>
	<boost/graph/connected_components.hpp>
	<boost/graph/depth_first_search.hpp>
	<boost/graph/filtered_graph.hpp>
	<boost/range/iterator_range.hpp>
	<vector>
)

add_subdirectory(test)
