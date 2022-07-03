cmake_minimum_required(VERSION 2.8.12)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
${PROJECT_SOURCE_DIR}/include
)

include_directories(
/usr/include/eigen3
)

add_executable(main pso.cpp main.cpp)
