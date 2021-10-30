cmake_minimum_required(VERSION 3.21)

file(GLOB tests *.cpp)

add_executable(tsplp-test ${tests})

target_link_libraries(tsplp-test PRIVATE tsplp)
