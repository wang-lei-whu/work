project(CUBE)
cmake_minimum_required(VERSION 3.19) ##仅3.19版本以上支持此类cuda语法
enable_language(CXX CUDA) # 采用cuda支持
# 定义执行文件和库文件输出路径
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



file(GLOB_RECURSE CURRENT_HEADERS *.cuh)
file(GLOB CURRENT_SOURCES *.cu)
 
# list(APPEND CUDA_NVCC_FLAGS "-std=c++11")


source_group("Include" FILES ${CURRENT_HEADERS}) # 包含目录下所有的cuh文件
source_group("Source" FILES ${CURRENT_SOURCES}) # 包含所有cu文件

add_library(gpu_test SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})

find_package(CUDA)
target_link_libraries(gpu_test ${CUDA_cublas_LIBRARY})

# 静态库
# 引入需要的头文件
# include_directories(
# ${PROJECT_SOURCE_DIR}/include
# )
# include_directories(
# /usr/include/eigen3
# )
# include_directories(
# /usr/local/cuda-11.6/include
# )
# 加载第三方库，暂无
# link_directories(/usr/local/cuda/lib)
# 编译本地库
# add_library(pso SHARED tools.cpp pso.cpp) # 动态库
# SET_TARGET_PROPERTIES(pso PROPERTIES VERSION 1.0 SOVERSION 1)

# add_library(pso_static STATIC pso.cpp) # 静态库
# SET_TARGET_PROPERTIES(pso_static PROPERTIES OUTPUT_NAME "pso")

# 编译并链接主程序
# link_directories(${PROJECT_SOURCE_DIR}/lib) #本地库路径，放在addadd_executable前
