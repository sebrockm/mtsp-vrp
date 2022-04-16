cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.hpp)
file(GLOB sources src/*.cpp src/*.hpp)

add_library(tsplp OBJECT ${headers} ${sources})

target_include_directories(tsplp PUBLIC include)

target_link_libraries(tsplp
	PRIVATE CONAN_PKG::coin-clp CONAN_PKG::boost
	PUBLIC CONAN_PKG::xtensor)

target_precompile_headers(tsplp
	PUBLIC
		<chrono>
		<cstddef>
		<exception>
		<limits>
		<memory>
		<span>
		<tuple>
		<unordered_map>
		<vector>
		<xtensor/xtensor.hpp>
	PRIVATE
		<ClpSimplex.hpp>
		<CoinFinite.hpp>

		<boost/graph/adjacency_list.hpp>
		<boost/graph/adjacency_matrix.hpp>
		<boost/graph/boykov_kolmogorov_max_flow.hpp>
		<boost/graph/connected_components.hpp>
		<boost/graph/depth_first_search.hpp>
		<boost/graph/filtered_graph.hpp>
		<boost/graph/one_bit_color_map.hpp>
		<boost/graph/stoer_wagner_min_cut.hpp>
		<boost/graph/topological_sort.hpp>
		<boost/graph/transitive_closure.hpp>
		<boost/property_map/function_property_map.hpp>
		<boost/range/adaptor/reversed.hpp>
		<boost/range/iterator_range.hpp>

		<xtensor/xadapt.hpp>
		<xtensor/xarray.hpp>
		<xtensor/xindex_view.hpp>
		<xtensor/xmanipulation.hpp>
		<xtensor/xvectorize.hpp>
		<xtensor/xview.hpp>

		<algorithm>
		<cassert>
		<cmath>
		<condition_variable>
		<future>
		<limits>
		<mutex>
		<optional>
		<stdexcept>
		<thread>
		<unordered_set>
)

add_subdirectory(test)
