cmake_minimum_required(VERSION 3.10)

# 项目名称
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 定义 cpp_helper 目录变量
set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/onnxruntime)

# 包含头文件目录
include_directories(${ONNXRUNTIME_DIR}/include_ort)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加库文件目录
link_directories(${ONNXRUNTIME_DIR}/lib)

# 打印源文件路径，确保找到src中的文件
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
message(STATUS "Source files found: ${SRC_FILES}")

# 如果没有找到文件，显示错误信息并停止配置
if(NOT SRC_FILES)
    message(FATAL_ERROR "No source files found in ${PROJECT_SOURCE_DIR}/src")
endif()

# 添加可执行文件
add_executable(demo demo.cpp ${SRC_FILES})

# 确保所有文件被正确编译
set(SOURCE_FILES demo.cpp ${SRC_FILES})

# 添加链接步骤
target_link_libraries(demo ${OpenCV_LIBS})
target_link_libraries(demo ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)

# 打印调试信息，确认链接的源文件
message(STATUS "Executable 'demo' will be linked with the following files: ${SOURCE_FILES}")
