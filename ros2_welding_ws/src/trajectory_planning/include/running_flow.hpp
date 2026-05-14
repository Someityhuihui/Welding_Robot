// ============================================
// running_flow.hpp - ROS2版本
// 修改点：所有ROS1头文件替换为ROS2版本
// ============================================

#ifndef RUNNING_FLOW_HPP
#define RUNNING_FLOW_HPP

#include <rclcpp/rclcpp.hpp>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>
#include <dirent.h>

#include <seam_location.hpp>
#include <motion_planning.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <std_msgs/msg/bool.hpp>

#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>

// 定义点云类型
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud; 
typedef pcl::PointCloud<pcl::PointXYZRGBL> PointCloudL;  
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::Normal> Normal;

// 外部变量声明
extern cv::Mat color_pic, depth_pic;
extern double camera_factor;
extern double camera_cx;
extern double camera_cy;  
extern double camera_fx;
extern double camera_fy;

// 注意：ROS2中不使用全局tf对象，改为参数传递
// extern tf::TransformListener listener;
// extern tf::TransformBroadcaster tf_broadcaster;

using namespace cv;
using namespace std;
using namespace Eigen;

// ============================================
// 函数声明（ROS2版本，发布者使用SharedPtr）
// ============================================

vector<geometry_msgs::msg::Pose> CAD_TrajectoryPlanning(
    Cloud::Ptr cloud_ptr,
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher, 
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime);

vector<geometry_msgs::msg::Pose> trajectory_6DOF_generation(
    pcl::PointXYZ realsense_position, 
    Cloud::Ptr cloud_ptr, 
    Cloud::Ptr cloud_ptr_modelSeam, 
    bool& trajectoryPlanning_flag, 
    int& receive_capture_count,
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher, 
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime);

bool welding_seam_location(
    Cloud::Ptr cloud_ptr, 
    pcl::PointXYZ realsense_position, 
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

vector<geometry_msgs::msg::Pose> trajectory_planning(
    Cloud::Ptr cloud_ptr_modelSeam,
    vector<pcl::PointXYZ> all_realsense_position,
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher,
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime);

void input_pointcloud_filter(Cloud::Ptr cloud_ptr);

void integrate_allsingle_pointcloudFrame(
    Cloud::Ptr cloud_ptr, 
    Cloud::Ptr cloud_ptr_modelSeam, 
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

int count_pointcloud_frameNum(string dataset_folder_path);

pcl::PointXYZ read_realtime_pointcloud_frame(
    string dataset_folder_path,
    int pointcloud_frameNum,
    int receive_capture_count,
    int& process_frame_count,
    bool& trajectoryPlanning_flag,
    Cloud::Ptr cloud_ptr);

void build_model_pointcloud(
    string dataset_folder_path, 
    int pointcloud_frameNum,
    PointCloud::Ptr model_pointcloud);

vector<geometry_msgs::msg::Pose> read_trajectory_frame(string trajectoryInfo_folder_path);

void show_pointcloud_Rviz(
    int show_Pointcloud_timeMax, 
    PointCloud::Ptr show_Rviz_cloud, 
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

void publish_pointcloud_Rviz(
    string coordinate, 
    PointCloud::Ptr pointloud, 
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

bool processing_frame_ornot(
    Cloud::Ptr cloud_ptr, 
    int show_Pointcloud_timeMax, 
    PointCloud::Ptr cloud_ptr_show, 
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

#endif // RUNNING_FLOW_HPP
