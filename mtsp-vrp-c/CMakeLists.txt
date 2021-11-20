cmake_minimum_required(VERSION 3.21)

file(GLOB headers include/*.h)
file(GLOB sources src/*.cpp)

add_library(mtsp-vrp-c SHARED ${headers} ${sources})

target_include_directories(mtsp-vrp-c PUBLIC include)

target_link_libraries(mtsp-vrp-c PRIVATE tsplp)

