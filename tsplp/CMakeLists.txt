cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.hpp)
file(GLOB sources src/*.cpp src/*.hpp)

add_library(tsplp STATIC ${headers} ${sources})

target_include_directories(tsplp PUBLIC include)

find_package(Threads REQUIRED)

target_link_libraries(tsplp
	PRIVATE graph-algos coin-clp::coin-clp Boost::Boost coverage_config
	PUBLIC xtensor::xtensor Threads::Threads)

target_precompile_headers(tsplp
	PUBLIC
	<chrono>
	<cstddef>
	<exception>
	<functional>
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

	<boost/core/bit.hpp>
	<boost/graph/adjacency_list.hpp>
	<boost/graph/adjacency_matrix.hpp>
	<boost/graph/boykov_kolmogorov_max_flow.hpp>
	<boost/graph/connected_components.hpp>
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
	<limits>
	<mutex>
	<optional>
	<span>
	<stdexcept>
	<thread>
	<unordered_set>
)

add_subdirectory(test)
