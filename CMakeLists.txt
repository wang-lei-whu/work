cmake_minimum_required(VERSION 2.8.12)
project(PSO)

add_subdirectory(srccuda)
add_subdirectory(src)

SET(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
${PROJECT_SOURCE_DIR}/include
)
include_directories(
/usr/include/eigen3
)

set(EXTRA_LIBS ${EXTRA_LIBS} gpu_test pso)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(main main.cpp) # 编译可执行程序
TARGET_LINK_LIBRARIES(main ${EXTRA_LIBS}) # 链接库文件
