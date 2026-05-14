// ============================================
// transformation.hpp - ROS2 最终正确版
// ============================================

#ifndef TRANSFORMATION_HPP
#define TRANSFORMATION_HPP

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <dirent.h>

// TF2 (ROS2 only)
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// ROS2 消息
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>

// Image & OpenCV
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// PCL 类型别名
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGBL> PointCloudL;
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::Normal> Normal;

// 外部全局变量
extern cv::Mat color_pic, depth_pic;
extern double camera_factor;
extern double camera_cx;
extern double camera_cy;
extern double camera_fx;
extern double camera_fy;

#include <opencv2/core.hpp>

// ============================================
// 函数声明（ROS2 规范 + 已修复所有错误）
// ============================================

pcl::PointXYZ camera_to_base_transform(const geometry_msgs::msg::TransformStamped& transform,
                                       pcl::PointXYZ Cam_Object);

pcl::PointXYZ realsense_position_acquisition(const geometry_msgs::msg::TransformStamped& transform);

void analyze_realsense_data(PointCloud::Ptr cloud);

void coordinate_transformation(const geometry_msgs::msg::TransformStamped& transform,
                               PointCloud::Ptr camera_pointcloud,
                               PointCloud::Ptr map_pointcloud,
                               Cloud::Ptr cloud_ptr);

geometry_msgs::msg::Pose Torch_to_End_transform(const geometry_msgs::msg::TransformStamped& transform_tool02torch);

void Base_to_End_transform(int& receive_pose_flag,
                           const geometry_msgs::msg::TransformStamped& transform);

Cloud::Ptr cloud_ptr_origin_copy(Cloud::Ptr cloud_ptr_new);

void input_pointcloud_filter(int process_count, int process_count_limit,
                             Cloud::Ptr cloud_ptr, Cloud::Ptr cloud_ptr_filter);


std::vector<geometry_msgs::msg::Pose> Ultimate_6DOF_TrajectoryGeneration(
    std::vector<geometry_msgs::msg::Pose>& Welding_Trajectory,
    Cloud::Ptr PathPoint_Position,
    std::vector<cv::Point3f> Torch_Normal_Vector);

tf2::Transform Waypoint_markerTransform_creation(int i, const geometry_msgs::msg::Pose& P);

std::string Waypoint_markerName_creation(int i);

void rotate_z(float x, float y, float z, float angle, float* x_output, float* y_output, float* z_output);
void rotate_x(float x, float y, float z, float angle, float* x_output, float* y_output, float* z_output);
void rotate_y(float x, float y, float z, float angle, float* x_output, float* y_output, float* z_output);

void euler_to_quaternion(float Yaw, float Pitch, float Roll, float Q[4]);

Eigen::Quaterniond rotation_Quaternionslerp(Eigen::Quaterniond starting,
                                            Eigen::Quaterniond ending,
                                            float t);

#endif // TRANSFORMATION_HPP
