# ============================================
# CMakeLists.txt - ROS2版本（集成PC2R算法）
# ============================================

cmake_minimum_required(VERSION 3.8)
project(trajectory_planning)

# ============================================
# 编译选项
# ============================================
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================
# 查找依赖包
# ============================================
find_package(ament_cmake REQUIRED)

# ROS2 核心包
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)

# TF2相关
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# 第三方库
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization kdtree filters features search)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem)

# 查找 OpenMP（可选，找不到也不报错）
find_package(OpenMP QUIET)

# ============================================
# 包含目录
# ============================================
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${visualization_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# ============================================
# 链接目录
# ============================================
link_directories(${PCL_LIBRARY_DIRS})

# ============================================
# 编译定义
# ============================================
add_definitions(${PCL_DEFINITIONS})
add_definitions(-D_USE_MATH_DEFINES)

# ============================================
# 可选：启用 C++14/17 标准
# ============================================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# 添加可执行文件（包含新算法）
# ============================================
add_executable(trajectory_planning_node 
  src/main.cpp
  src/running_flow.cpp
  src/motion_planning.cpp
  src/transformation.cpp
  src/seam_location.cpp
  src/pc2r_welding_generator.cpp
  src/adaptive_pointcloud_processor.cpp
  src/welding_config.cpp
)

# ============================================
# 设置编译选项（在 add_executable 之后）
# ============================================
if(OpenMP_CXX_FOUND)
  target_compile_options(trajectory_planning_node PRIVATE ${OpenMP_CXX_FLAGS})
  message(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
else()
  message(STATUS "OpenMP not found, using single-threaded mode")
endif()

target_compile_options(trajectory_planning_node PRIVATE -O3)

# ============================================
# 链接库
# ============================================
target_link_libraries(trajectory_planning_node
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${image_transport_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${tf2_geometry_msgs_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenMP_CXX_LIBRARIES}
)

# ============================================
# ament 目标依赖
# ============================================
ament_target_dependencies(trajectory_planning_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  cv_bridge
  image_transport
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# ============================================
# 安装规则
# ============================================
install(TARGETS trajectory_planning_node
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================
# 必须的 ament 宏
# ============================================
ament_package()
