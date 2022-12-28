cmake_minimum_required(VERSION 3.21)

set(demo_script "${CMAKE_CURRENT_BINARY_DIR}/demo_script.sh")

file(GENERATE
	OUTPUT "${demo_script}"
	CONTENT "\
\"${Python3_EXECUTABLE}\" -m venv \"${testenv}\"\n\
${activate_command}\n\
pip install -r \"${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\"\n\
python \"${CMAKE_CURRENT_SOURCE_DIR}/progress_visualization.py\"\n\
"
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

add_custom_target(progress-visualization
	COMMENT "setting up virtual python environment and running demos"
	COMMAND sh ${demo_script}
	VERBATIM
	USES_TERMINAL
	SOURCES requirements.txt progress_visualization.py
)

add_dependencies(progress-visualization mtsp-vrp-python-package)

add_test(
	NAME progress-visualization-test
	COMMAND sh ${demo_script}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
