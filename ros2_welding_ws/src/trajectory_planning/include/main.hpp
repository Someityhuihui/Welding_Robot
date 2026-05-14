// ============================================
// main.hpp - ROS2版本
// 原文件：main.h
// 修改点：所有ROS1头文件替换为ROS2版本
// ============================================

#ifndef MAIN_HPP
#define MAIN_HPP

// ============================================
// 修改点1: ROS2 核心头文件
// ============================================
#include <rclcpp/rclcpp.hpp>

#include <math.h>
#include <iostream>   
#include <vector>
#include <string>
#include <dirent.h>

// ============================================
// 修改点2: 项目内部头文件（需要对应修改）
// ============================================
#include <running_flow.hpp>
#include <seam_location.hpp>
#include <motion_planning.hpp>
#include <transformation.hpp>

// ============================================
// 修改点3: ROS2 消息头文件
// ============================================
#include <std_msgs/msg/string.hpp>

// ============================================
// 修改点4: 图像处理（ROS2版本）
// ============================================
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>

// ============================================
// 修改点5: TF2（替代tf）
// ============================================
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <std_msgs/msg/bool.hpp>
#include <visualization_msgs/msg/marker.hpp>

// ============================================
// 修改点6: PCL库（保持不变）
// ============================================
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// ============================================
// 修改点7: 类型定义（保持不变）
// ============================================
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud; 
typedef pcl::PointCloud<pcl::PointXYZRGBL> PointCloudL;  
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::Normal> Normal;

// ============================================
// 修改点8: 外部变量声明（ROS2中，这些需要从节点获取）
// 注意：在ROS2中，相机参数应该通过参数服务器获取
// ============================================
extern cv::Mat color_pic, depth_pic;
extern double camera_factor;
extern double camera_cx;
extern double camera_cy;  
extern double camera_fx;
extern double camera_fy;

// ============================================
// 修改点9: TF2相关（替代原来的tf）
// 注意：ROS2中不能直接 extern tf2_ros对象
// 建议通过函数参数传递或使用单例模式
// ============================================
// ROS1: extern tf::TransformListener listener;
// ROS1: extern tf::TransformBroadcaster tf_broadcaster;
// ROS2替代方案：
// 1. 通过函数参数传递 std::shared_ptr<tf2_ros::Buffer>
// 2. 或者使用全局指针（不推荐）

// 推荐：在需要使用tf的地方，通过参数传递
// 示例函数声明：
// void some_function(std::shared_ptr<tf2_ros::Buffer> tf_buffer);

#endif // MAIN_HPP
