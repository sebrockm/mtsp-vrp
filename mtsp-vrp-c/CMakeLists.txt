cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.h)
file(GLOB sources src/*.cpp)

add_library(mtsp-vrp-c SHARED ${headers} ${sources})

target_link_libraries(mtsp-vrp-c PRIVATE tsplp)

target_include_directories(mtsp-vrp-c
	PUBLIC
		include
		"${CMAKE_CURRENT_BINARY_DIR}" # GenerateExportHeader generates the header here
)

target_precompile_headers(mtsp-vrp-c
	PRIVATE
		<array>
		<chrono>
		<stddef.h>
		<xtensor/xadapt.hpp>
)

include(GenerateExportHeader)
generate_export_header(mtsp-vrp-c)
