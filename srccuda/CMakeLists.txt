project(CUBE)
cmake_minimum_required(VERSION 3.9) ##3.9版本以上支持此类cuda语法
enable_language(CXX CUDA) # 激活cmake的cuda支持
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../lib)


file(GLOB_RECURSE CURRENT_HEADERS *.cuh)
file(GLOB CURRENT_SOURCES *.cu)
 
# list(APPEND CUDA_NVCC_FLAGS "-std=c++11")


source_group("Include" FILES ${CURRENT_HEADERS}) # 包含目录下所有的cuh文件
source_group("Source" FILES ${CURRENT_SOURCES}) # 包含当前目录下所有cu文件

add_library(gpu_test SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})

find_package(CUDA)
target_link_libraries(gpu_test ${CUDA_cublas_LIBRARY})
