cmake_minimum_required(VERSION 3.10)
project(weld_vision_extractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 PCL（需要 segmentation 组件）
find_package(PCL REQUIRED COMPONENTS common io filters kdtree features segmentation registration)

# 查找 Eigen3
find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)

# 包含目录
include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# 链接目录
link_directories(${PCL_LIBRARY_DIRS})

# 定义 PCL 宏
add_definitions(${PCL_DEFINITIONS})

# 源文件
set(SOURCES
    src/pointcloud_registrar.cpp
    src/plane_extractor.cpp
    src/corner_classifier.cpp
    src/weld_extractor.cpp
)

# 添加可执行文件
add_executable(weld_extractor ${SOURCES})

# ✅ 正确链接 PCL 库
target_link_libraries(weld_extractor 
    ${PCL_LIBRARIES}
    ${EIGEN3_LIBRARIES}
)

# 如果启用 OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(weld_extractor PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(weld_extractor ${OpenMP_CXX_LIBRARIES})
    message(STATUS "OpenMP enabled")
endif()

# 安装
install(TARGETS weld_extractor DESTINATION bin)
install(FILES 
    include/pointcloud_registrar.hpp
    include/plane_extractor.hpp
    include/corner_classifier.hpp
    include/weld_extractor.hpp
    DESTINATION include
)
