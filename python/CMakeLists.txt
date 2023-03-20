cmake_minimum_required(VERSION 3.21)

include(FindPython3)

if (${Python3_FOUND})
	message("Found python3: ${Python3_EXECUTABLE}")
else()
	message(FATAL_ERROR "python3 was not found")
endif()

if (${Python3_VERSION} VERSION_LESS 3.4)
	message(FATAL_ERROR "python 3.4 or higher is required")
endif()

set(testenv "${CMAKE_CURRENT_BINARY_DIR}/testenv")
if (WIN32)
	set(activate_command "\"${testenv}/Scripts/activate.bat\"")
else()
	set(activate_command "source \"${testenv}/bin/activate\"")
endif()
set(mtsp_vrp_c_lib_path_file "${CMAKE_CURRENT_SOURCE_DIR}/_mtsp_vrp_c_lib_path.txt")

add_custom_command(
	OUTPUT ${mtsp_vrp_c_lib_path_file}
	COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_FILE:mtsp-vrp-c>" > ${mtsp_vrp_c_lib_path_file}
)

add_custom_target(mtsp-vrp-python-package ALL
	COMMENT "Creating python package"
	DEPENDS ${mtsp_vrp_c_lib_path_file}
	SOURCES mtsp_vrp.py
)

add_dependencies(mtsp-vrp-python-package mtsp-vrp-c)

add_subdirectory(demos)
add_subdirectory(tsplib)
