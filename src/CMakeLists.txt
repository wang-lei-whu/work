project(PSO)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_C_COMPILER g++)
list(APPEND CMAKE_C_FLAGS "lstdc++")
SET(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# 定义执行文件和库文件输出路径
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../lib)

# 引入需要的头文件
include_directories(
${PROJECT_SOURCE_DIR}/../include
)
include_directories(
/usr/include/eigen3
)


# 编译本地库
add_library(pso SHARED tools.cpp pso.cpp) # 动态库
SET_TARGET_PROPERTIES(pso PROPERTIES VERSION 1.0 SOVERSION 1)
