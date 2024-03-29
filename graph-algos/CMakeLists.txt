cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.hpp)
file(GLOB sources src/*.cpp src/*.hpp)

add_library(graph-algos STATIC ${headers} ${sources})

target_include_directories(graph-algos PUBLIC include)

target_link_libraries(graph-algos PUBLIC Boost::Boost PRIVATE coverage_config)

target_precompile_headers(graph-algos
	PUBLIC
	<boost/graph/depth_first_search.hpp>
	<boost/graph/graph_traits.hpp>
	<boost/graph/undirected_dfs.hpp>
	<boost/property_map/property_map.hpp>

	<functional>
	<span>
	<unordered_map>

	PRIVATE
	<boost/graph/adjacency_list.hpp>
	<boost/graph/boykov_kolmogorov_max_flow.hpp>
	<boost/graph/connected_components.hpp>
	<boost/graph/filtered_graph.hpp>
	<boost/graph/properties.hpp>
	<boost/iterator/counting_iterator.hpp>
	<boost/iterator/filter_iterator.hpp>
	<boost/iterator/transform_iterator.hpp>
	<boost/property_map/function_property_map.hpp>
	<boost/range/iterator_range.hpp>

	<limits>
	<utility>
	<vector>
)

add_subdirectory(test)
