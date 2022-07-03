cmake_minimum_required(VERSION 2.8.12)
project(PSO)
	
SET(CMAKE_BUILD_TYPE debug)
add_subdirectory(src)
#设置debug模式
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
