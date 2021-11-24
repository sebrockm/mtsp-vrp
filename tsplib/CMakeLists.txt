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
	set(activate_command "${testenv}/Scripts/activate.bat")
else()
	set(activate_command "source ${testenv}/bin/activate")
endif()

add_custom_target(tsplib-bench ALL
	COMMENT "setting up python environment and running tsplib.py"
	COMMAND "${Python3_EXECUTABLE}" -m venv "${testenv}"
	COMMAND ls ${testenv}/bin
	COMMAND ${activate_command}
	COMMAND pip install -r "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
	COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/tsplib.py" "$<TARGET_FILE:mtsp-vrp-c>" "${CMAKE_CURRENT_SOURCE_DIR}"
	COMMAND deactivate
	VERBATIM
	USES_TERMINAL
	SOURCES tsplib.py requirements.txt)
