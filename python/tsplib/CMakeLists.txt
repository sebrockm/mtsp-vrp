cmake_minimum_required(VERSION 3.21)

set(bench_script "${CMAKE_CURRENT_BINARY_DIR}/bench_script.sh")

file(GENERATE
	OUTPUT "${bench_script}"
	CONTENT "\
\"${Python3_EXECUTABLE}\" -m venv --system-site-packages \"${testenv}\"\n\
${activate_command}\n\
pip install -r \"${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\"\n\
python \"${CMAKE_CURRENT_SOURCE_DIR}/tsplib.py\" 60000\n\
"
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

add_custom_target(tsplib-bench
	COMMENT "setting up virtual python environment and running tsplib.py"
	COMMAND "${bench_script}"
	VERBATIM
	USES_TERMINAL
	SOURCES best-known-solutions.json readme.md requirements.txt tsplib.py
)

add_dependencies(tsplib-bench mtsp-vrp-python-package)
