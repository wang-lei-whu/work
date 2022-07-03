cmake_minimum_required(VERSION 2.8.12)
project(PSO)

add_subdirectory(src)
#设置debug
SET(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
