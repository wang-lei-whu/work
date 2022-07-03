cmake_minimum_required(VERSION 2.8.12)
# 定义执行文件和库文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 引入需要的头文件
include_directories(
${PROJECT_SOURCE_DIR}/include
)
include_directories(
/usr/include/eigen3
)
# 加载第三方库，暂无

# 编译本地库
add_library(pso SHARED tools.cpp pso.cpp) # 动态库
SET_TARGET_PROPERTIES(pso PROPERTIES VERSION 1.0 SOVERSION 1)

# add_library(pso_static STATIC pso.cpp) # 静态库
# SET_TARGET_PROPERTIES(pso_static PROPERTIES OUTPUT_NAME "pso")

# 编译并链接主程序
link_directories(${PROJECT_SOURCE_DIR}/lib) #本地库路径，放在addadd_executable前

add_executable(main main.cpp) # 编译可执行程序
TARGET_LINK_LIBRARIES(main pso) # 链接可执行程序和库文件
