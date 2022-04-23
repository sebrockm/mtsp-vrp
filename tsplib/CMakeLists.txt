cmake_minimum_required(VERSION 3.21)

include(FindPython3)

if (${Python3_FOUND})
	message("Found python3: ${Python3_EXECUTABLE}")
else()
	message(FATAL_ERROR "python3 was not found, aborting tsplib test")
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

set(bench_script "${CMAKE_CURRENT_BINARY_DIR}/bench_script.sh")

file(GENERATE
	OUTPUT ${bench_script}
	CONTENT "\
\"${Python3_EXECUTABLE}\" -m venv \"${testenv}\"\n\
${activate_command}\n\
pip install -r \"${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\"\n\
python \"${CMAKE_CURRENT_SOURCE_DIR}/tsplib.py\" \"$<TARGET_FILE:mtsp-vrp-c>\" 60000"
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

add_custom_target(tsplib-bench
	COMMENT "setting up virtual python environment and running tsplib.py"
	COMMAND "${bench_script}"
	VERBATIM
	USES_TERMINAL
	SOURCES requirements.txt tsplib.py tsplib/best-known-solutions.json
)
