cmake_minimum_required(VERSION 3.21)

file(GLOB tests *.cpp)

add_executable(tsplp-test ${tests})

target_link_libraries(tsplp-test PRIVATE tsplp CONAN_PKG::catch2)

target_precompile_headers(tsplp-test
	PRIVATE
	<catch2/catch.hpp>
	<vector>
)

set_source_files_properties(catch-main.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

add_test(NAME tsplp-test COMMAND tsplp-test)
