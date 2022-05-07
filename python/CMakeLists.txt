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

file(GENERATE
	OUTPUT ${mtsp_vrp_c_lib_path_file}
	CONTENT "$<TARGET_FILE:mtsp-vrp-c>"
)

add_custom_target(mtsp-vrp-python-package ALL
	COMMENT "Creating python package"
	DEPENDS ${mtsp_vrp_c_lib_path_file}
	SOURCES mtsp_vrp.py
)

add_dependencies(mtsp-vrp-python-package mtsp-vrp-c)

set(test_script "${CMAKE_CURRENT_BINARY_DIR}/test_script.sh")

file(GENERATE
	OUTPUT "${test_script}"
	CONTENT "\
\"${Python3_EXECUTABLE}\" -m venv \"${testenv}\"\n\
${activate_command}\n\
python -c \"import mtsp_vrp; assert(mtsp_vrp.solve_mtsp_vrp);\"\n
"
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

add_test(NAME package-test COMMAND "${test_script}")

add_subdirectory(tsplib)
