// ============================================
// main.cpp - ROS2版本（带Marker可视化）
// ============================================

#include <tf2_ros/buffer.h>
#include <rclcpp/rclcpp.hpp>

#include "main.hpp"
#include "pc2r_welding_generator.hpp"
#include "welding_config.hpp"

#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <ctime>

// ROS2 消息头文件
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <visualization_msgs/msg/marker.hpp>

// ROS2 图像处理
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>

// ROS2 TF2
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// PCL库
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

using namespace std;
using namespace cv;

// ============================================
// 类型定义
// ============================================
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;

// ============================================
// PC2R 处理结果结构体
// ============================================
struct PC2RResult {
    std::vector<geometry_msgs::msg::Pose> trajectory;
    Cloud::Ptr edge_points;      // 焊缝边缘点（红色）
    Cloud::Ptr path_points;      // B样条路径点（绿色）
    Cloud::Ptr stitched_cloud;   // 拼接后的完整点云（浅灰色）
    Cloud::Ptr wsr_cloud;        // 焊缝区域点云（蓝色）- 新增
    Cloud::Ptr aligned_cloud;    // ICP配准结果点云（橙色）- 新增
};

// ============================================
// 全局变量
// ============================================
static bool algorithm_selected = false;
static bool use_pc2r = false;

// 相机内参
double camera_factor = 1000;
double camera_cx = 311.2325744628906;
double camera_cy = 226.9261474609375;
double camera_fx = 619.9661254882812;
double camera_fy = 619.856201171875;

// 控制标志
bool trajectoryPlanning_flag = false;
int receive_pose_flag = 0, process_count = 0, process_count_limit = 1;
float current_x = 0, current_y = 0, current_z = 0;
float current_yaw = 0, current_pitch = 0, current_roll = 0;

// 图像数据
cv_bridge::CvImagePtr color_ptr, depth_ptr;
cv::Mat color_pic, depth_pic;

int receive_capture_count = 1;
int process_frame_count = 1;

// ============================================
// 函数声明
// ============================================

int count_pointcloud_frameNum(string dataset_folder_path);
void build_model_pointcloud(string dataset_folder_path, int pointcloud_frameNum, PointCloud::Ptr model_pointcloud_display);
vector<geometry_msgs::msg::Pose> trajectory_6DOF_generation(
    bool read_realtime_result,
    Cloud::Ptr cloud_ptr,
    Cloud::Ptr cloud_ptr_modelSeam,
    bool& trajectoryPlanning_flag,
    int& receive_capture_count,
    sensor_msgs::msg::PointCloud2 pub_pointcloud,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher,
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime
);
void publish_pointcloud_Rviz(string coordinate, PointCloud::Ptr pointloud, 
                             sensor_msgs::msg::PointCloud2 pub_pointcloud,
                             rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);
tf2::Transform create_waypoint_transform(int i, geometry_msgs::msg::Pose pose);
pcl::PointXYZ read_realtime_pointcloud_frame(string dataset_folder_path,
                                              int pointcloud_frameNum,
                                              int receive_capture_count,
                                              int& process_frame_count,
                                              bool& trajectoryPlanning_flag,
                                              Cloud::Ptr cloud_ptr);

// PC2R 算法轨迹生成函数
// ============================================
// 修改后的函数声明 - 返回结构体
// ============================================
PC2RResult trajectory_6DOF_generation_pc2r(
    const std::vector<Cloud::Ptr>& multi_view_clouds,
    const std::vector<Eigen::Vector3f>& camera_poses,
    Cloud::Ptr cloud_ptr_modelSeam,
    bool& trajectoryPlanning_flag,
    int& receive_capture_count,
    sensor_msgs::msg::PointCloud2 pub_pointcloud,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher,
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime,
    rclcpp::Logger logger);

// ============================================
// create_waypoint_transform 实现
// ============================================
tf2::Transform create_waypoint_transform(int i, geometry_msgs::msg::Pose pose)
{
    (void)i;  // 消除未使用参数警告
    tf2::Transform transform;
    tf2::Vector3 origin(pose.position.x, pose.position.y, pose.position.z);
    tf2::Quaternion rotation(pose.orientation.x, pose.orientation.y, 
                              pose.orientation.z, pose.orientation.w);
    transform.setOrigin(origin);
    transform.setRotation(rotation);
    return transform;
}

// 回调函数声明
void color_Callback(const sensor_msgs::msg::Image::SharedPtr color_msg);
void depth_Callback(const sensor_msgs::msg::Image::SharedPtr depth_msg);
void pointcloud_storageFolder_Callback(const std_msgs::msg::String::SharedPtr msg);


// ============================================
// 初始化 PC2R 生成器的函数
// ============================================
// main.cpp 中的 configurePC2RGenerator 函数

void configurePC2RGenerator(PC2RWeldingGenerator& generator) {
    // 使用全局配置
    AdaptiveConfig adaptive_config;
    adaptive_config.target_points_min = g_config.downsample.target_points_min;
    adaptive_config.target_points_max = g_config.downsample.target_points_max;
    adaptive_config.voxel_size_min = g_config.downsample.voxel_size_min;
    adaptive_config.voxel_size_max = g_config.downsample.voxel_size_max;
    adaptive_config.bbox_change_ratio = g_config.stitching.bbox_change_threshold;
    adaptive_config.min_overlap_ratio = g_config.stitching.min_overlap_ratio;
    adaptive_config.icp_fitness_threshold = g_config.stitching.icp_fitness_threshold;
    adaptive_config.max_icp_iterations = g_config.stitching.icp_max_iterations;
    adaptive_config.max_correspondence_distance = g_config.stitching.max_correspondence_distance;
    adaptive_config.ransac_iterations = g_config.stitching.ransac_iterations;
    
    generator.setAdaptiveConfig(adaptive_config);
    generator.enableAdaptiveProcessing(g_config.perf.enable_adaptive_processing);
    
    // 使用配置中的参数
    generator.initialize(
        g_config.stitching.ibnn_epsilon,
        g_config.edge.search_radius,
        g_config.edge.epsilon,
        g_config.bspline.degree
    );
    
    // 设置边缘检测参数
    generator.setEdgeDetectionParams(
        g_config.edge.search_radius,
        g_config.edge.epsilon,
        g_config.edge.intensity_threshold,
        g_config.edge.min_neighbors
    );
    
    // 设置WSR提取参数
    generator.setWSRParams(g_config.wsr.cylinder_radius);
    
    // 设置路径排序参数
    generator.setSortingParams(g_config.sorting.neighbor_radius);
}


// ============================================
// main函数
// ============================================
int main(int argc, char **argv)
{
    // 加载配置
    loadConfig();
    printConfig();
    
    // 初始化ROS2节点
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("trajectory_planning");
    auto logger = node->get_logger();
    rclcpp::Rate naptime(1000);
    
    // ============================================
    // 创建订阅者和发布者
    // ============================================
    
    auto sub = node->create_subscription<std_msgs::msg::String>(
        "pointcloud_storageFolder", 10, pointcloud_storageFolder_Callback);
    auto color_sub = node->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_raw", 10, color_Callback);
    auto depth_sub = node->create_subscription<sensor_msgs::msg::Image>(
        "/camera/aligned_depth_to_color/image_raw", 10, depth_Callback);
    
    auto Moveit_path_publisher = node->create_publisher<geometry_msgs::msg::Pose>("Moveit_motion_Path", 10);
    auto Welding_Trajectory_publisher = node->create_publisher<geometry_msgs::msg::Pose>("Welding_Trajectory", 10);
    auto pointcloud_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("processing_pointcloud", 10);
    auto model_pointcloud_display_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("model_pointcloud_display", 10);
    auto vis_pub = node->create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 10);
    
    // ============================================
    // TF2
    // ============================================
    auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    auto tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);
    
    // ============================================
    // 点云消息和指针
    // ============================================
    sensor_msgs::msg::PointCloud2 pub_model_pointcloud_display;
    sensor_msgs::msg::PointCloud2 pub_path;
    sensor_msgs::msg::PointCloud2 pub_pointcloud;
    
    PointCloud::Ptr camera_pointcloud(new PointCloud);
    PointCloud::Ptr cam_pc_transform(new PointCloud);
    PointCloud::Ptr model_pointcloud_display(new PointCloud);
    Cloud::Ptr cloud_ptr(new Cloud);
    Cloud::Ptr cloud_ptr_modelSeam(new Cloud);
    
    // ============================================
    // 读取点云数据集
    // ============================================
    int pointcloud_frameNum = count_pointcloud_frameNum(g_config.data_path.dataset_folder);
    build_model_pointcloud(g_config.data_path.dataset_folder, pointcloud_frameNum, model_pointcloud_display);

    // ============================================
    // 算法选择
    // ============================================
    if (!algorithm_selected) {
        cout << endl;
        cout << "========================================" << endl;
        cout << "Select algorithm:" << endl;
        cout << "  1 - Original algorithm (single frame)" << endl;
        cout << "  2 - PC2R algorithm (multi-view stitching + adaptive processing)" << endl;
        cout << "========================================" << endl;
        cout << "Enter choice (1 or 2): ";
        int choice;
        cin >> choice;
        use_pc2r = (choice == 2);
        algorithm_selected = true;
        
        if (use_pc2r) {
            cout << "Using PC2R (Point Cloud to Robotic Welding) Algorithm..." << endl;
            cout << "Features: Adaptive downsampling, keyframe detection, incremental stitching" << endl;
        } else {
            cout << "Using Original algorithm..." << endl;
        }
    }
    
    // ============================================
    // 主循环
    // ============================================
    vector<geometry_msgs::msg::Pose> Rviz_TrajectoryPose;

    while (rclcpp::ok())
    {
        // ===================== PC2R 模式 =====================
        if (use_pc2r) {
            // 【修复】所有变量提到作用域顶部，确保全局可见
            static std::vector<Cloud::Ptr> multi_view_buffer;
            static std::vector<Eigen::Vector3f> pose_buffer;
            static bool processing_done = false;
            static int max_frames = g_config.acquisition.max_frames;
            static bool collecting = true;
            static bool rviz_guide_shown = false;
            // static Cloud::Ptr processing_cloud;
            static pcl::PointCloud<pcl::PointXYZ>::Ptr processing_cloud;
            static PC2RResult pc2r_result;

            // main.cpp - 在 PC2R 模式的 processing_done 部分，替换现有的点云发布代码

            if (processing_done) {
                // ---------------------- 发布 TF ----------------------
                if (!Rviz_TrajectoryPose.empty()) {
                    for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++) {
                        string markerFrame = Waypoint_markerName_creation(i);
                        tf2::Transform waypoint_transform = create_waypoint_transform(i, Rviz_TrajectoryPose[i]);

                        geometry_msgs::msg::TransformStamped waypoint_Marker;
                        waypoint_Marker.header.stamp = node->get_clock()->now();
                        waypoint_Marker.header.frame_id = g_config.viz.fixed_frame;
                        waypoint_Marker.child_frame_id = markerFrame;
                        waypoint_Marker.transform.translation.x = waypoint_transform.getOrigin().x();
                        waypoint_Marker.transform.translation.y = waypoint_transform.getOrigin().y();
                        waypoint_Marker.transform.translation.z = waypoint_transform.getOrigin().z();
                        waypoint_Marker.transform.rotation.x = waypoint_transform.getRotation().x();
                        waypoint_Marker.transform.rotation.y = waypoint_transform.getRotation().y();
                        waypoint_Marker.transform.rotation.z = waypoint_transform.getRotation().z();
                        waypoint_Marker.transform.rotation.w = waypoint_transform.getRotation().w();
                        tf_broadcaster->sendTransform(waypoint_Marker);
                    }

                    // 发布轨迹
                    for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++) {
                        Welding_Trajectory_publisher->publish(Rviz_TrajectoryPose[i]);
                    }

                    // ---------------------- 发布 Marker ----------------------
                    for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++) {
                        // 箭头
                        visualization_msgs::msg::Marker marker;
                        marker.header.frame_id = g_config.viz.fixed_frame;
                        marker.header.stamp = node->get_clock()->now();
                        marker.ns = "waypoint_marker";
                        marker.id = i;
                        marker.type = visualization_msgs::msg::Marker::ARROW;
                        marker.action = visualization_msgs::msg::Marker::ADD;
                        marker.points.resize(2);
                        marker.points[0].x = Rviz_TrajectoryPose[i].position.x;
                        marker.points[0].y = Rviz_TrajectoryPose[i].position.y;
                        marker.points[0].z = Rviz_TrajectoryPose[i].position.z;
                        marker.points[1].x = marker.points[0].x + Rviz_TrajectoryPose[i].orientation.x * g_config.viz.marker_arrow_length;
                        marker.points[1].y = marker.points[0].y + Rviz_TrajectoryPose[i].orientation.y * g_config.viz.marker_arrow_length;
                        marker.points[1].z = marker.points[0].z + Rviz_TrajectoryPose[i].orientation.z * g_config.viz.marker_arrow_length;
                        marker.color.r = 1.0f;
                        marker.color.g = 0.0f;
                        marker.color.b = 0.0f;
                        marker.color.a = 1.0f;
                        marker.scale.x = 0.008f;
                        marker.scale.y = 0.015f;
                        marker.scale.z = 0.04f;
                        vis_pub->publish(marker);

                        // 球体
                        visualization_msgs::msg::Marker sphere_marker;
                        sphere_marker.header.frame_id = g_config.viz.fixed_frame;
                        sphere_marker.header.stamp = node->get_clock()->now();
                        sphere_marker.ns = "waypoint_sphere";
                        sphere_marker.id = i;
                        sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
                        sphere_marker.action = visualization_msgs::msg::Marker::ADD;
                        sphere_marker.pose.position.x = Rviz_TrajectoryPose[i].position.x;
                        sphere_marker.pose.position.y = Rviz_TrajectoryPose[i].position.y;
                        sphere_marker.pose.position.z = Rviz_TrajectoryPose[i].position.z;
                        sphere_marker.pose.orientation.w = 1.0f;
                        sphere_marker.color.r = 0.0f;
                        sphere_marker.color.g = 1.0f;
                        sphere_marker.color.b = 0.0f;
                        sphere_marker.color.a = 1.0f;
                        sphere_marker.scale.x = g_config.viz.marker_sphere_size;
                        sphere_marker.scale.y = g_config.viz.marker_sphere_size;
                        sphere_marker.scale.z = g_config.viz.marker_sphere_size;
                        vis_pub->publish(sphere_marker);
                    }
                }

                // ============================================
                // 完整的点云可视化（5种点云全部发布）
                // ============================================
                
                // 1. 发布模型点云（灰色/白色）- 静态显示一次即可
                static bool model_published = false;
                if (!model_published && model_pointcloud_display && !model_pointcloud_display->points.empty()) {
                    publish_pointcloud_Rviz(g_config.viz.fixed_frame, model_pointcloud_display,
                                            pub_model_pointcloud_display, model_pointcloud_display_publisher);
                    model_published = true;
                    RCLCPP_INFO(logger, "Published model point cloud: %zu points", model_pointcloud_display->points.size());
                }
                
                // 2. 发布拼接后的完整点云（浅灰色/彩色）
                if (pc2r_result.stitched_cloud && !pc2r_result.stitched_cloud->points.empty()) {
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_stitched(new pcl::PointCloud<pcl::PointXYZRGB>);
                    for (const auto& p : pc2r_result.stitched_cloud->points) {
                        pcl::PointXYZRGB rgb_p;
                        rgb_p.x = p.x;
                        rgb_p.y = p.y;
                        rgb_p.z = p.z;
                        rgb_p.r = 180;
                        rgb_p.g = 180;
                        rgb_p.b = 200;
                        rgb_stitched->push_back(rgb_p);
                    }
                    sensor_msgs::msg::PointCloud2 pub_stitched;
                    pcl::toROSMsg(*rgb_stitched, pub_stitched);
                    pub_stitched.header.frame_id = g_config.viz.fixed_frame;
                    pub_stitched.header.stamp = node->get_clock()->now();
                    pointcloud_publisher->publish(pub_stitched);
                    RCLCPP_INFO(logger, "Published stitched cloud: %zu points", rgb_stitched->points.size());
                }
                
                // 3. 发布WSR焊缝区域点云（蓝色）
                if (pc2r_result.wsr_cloud && !pc2r_result.wsr_cloud->points.empty()) {
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_wsr(new pcl::PointCloud<pcl::PointXYZRGB>);
                    for (const auto& p : pc2r_result.wsr_cloud->points) {
                        pcl::PointXYZRGB rgb_p;
                        rgb_p.x = p.x;
                        rgb_p.y = p.y;
                        rgb_p.z = p.z;
                        rgb_p.r = 0;
                        rgb_p.g = 100;
                        rgb_p.b = 255;
                        rgb_wsr->push_back(rgb_p);
                    }
                    sensor_msgs::msg::PointCloud2 pub_wsr;
                    pcl::toROSMsg(*rgb_wsr, pub_wsr);
                    pub_wsr.header.frame_id = g_config.viz.fixed_frame;
                    pub_wsr.header.stamp = rclcpp::Clock().now();
                    pointcloud_publisher->publish(pub_wsr);
                    RCLCPP_INFO(logger, "Published WSR cloud: %zu points (blue)", rgb_wsr->points.size());
                }
                
                // 4. 发布焊缝边缘点云（红色）
                if (pc2r_result.edge_points && !pc2r_result.edge_points->points.empty()) {
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_edge(new pcl::PointCloud<pcl::PointXYZRGB>);
                    for (const auto& p : pc2r_result.edge_points->points) {
                        pcl::PointXYZRGB rgb_p;
                        rgb_p.x = p.x;
                        rgb_p.y = p.y;
                        rgb_p.z = p.z;
                        rgb_p.r = 255;
                        rgb_p.g = 0;
                        rgb_p.b = 0;
                        rgb_edge->push_back(rgb_p);
                    }
                    sensor_msgs::msg::PointCloud2 pub_edge;
                    pcl::toROSMsg(*rgb_edge, pub_edge);
                    pub_edge.header.frame_id = g_config.viz.fixed_frame;
                    pub_edge.header.stamp = rclcpp::Clock().now();
                    pointcloud_publisher->publish(pub_edge);
                    RCLCPP_INFO(logger, "Published edge points: %zu points (red)", rgb_edge->points.size());
                }
                
                // 5. 发布B样条路径点（绿色/青色）
                if (pc2r_result.path_points && !pc2r_result.path_points->points.empty()) {
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_path(new pcl::PointCloud<pcl::PointXYZRGB>);
                    for (const auto& p : pc2r_result.path_points->points) {
                        pcl::PointXYZRGB rgb_p;
                        rgb_p.x = p.x;
                        rgb_p.y = p.y;
                        rgb_p.z = p.z;
                        rgb_p.r = 0;
                        rgb_p.g = 255;
                        rgb_p.b = 100;
                        rgb_path->push_back(rgb_p);
                    }
                    sensor_msgs::msg::PointCloud2 pub_path;
                    pcl::toROSMsg(*rgb_path, pub_path);
                    pub_path.header.frame_id = g_config.viz.fixed_frame;
                    pub_path.header.stamp = rclcpp::Clock().now();
                    pointcloud_publisher->publish(pub_path);
                    RCLCPP_INFO(logger, "Published path points: %zu points (green)", rgb_path->points.size());
                }
                
                // 6. 发布ICP配准对比可视化（可选）
                if (pc2r_result.aligned_cloud && !pc2r_result.aligned_cloud->points.empty()) {
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_aligned(new pcl::PointCloud<pcl::PointXYZRGB>);
                    for (const auto& p : pc2r_result.aligned_cloud->points) {
                        pcl::PointXYZRGB rgb_p;
                        rgb_p.x = p.x;
                        rgb_p.y = p.y;
                        rgb_p.z = p.z;
                        rgb_p.r = 255;
                        rgb_p.g = 200;
                        rgb_p.b = 0;
                        rgb_aligned->push_back(rgb_p);
                    }
                    sensor_msgs::msg::PointCloud2 pub_aligned;
                    pcl::toROSMsg(*rgb_aligned, pub_aligned);
                    pub_aligned.header.frame_id = g_config.viz.fixed_frame;
                    pub_aligned.header.stamp = rclcpp::Clock().now();
                    pointcloud_publisher->publish(pub_aligned);
                    RCLCPP_INFO(logger, "Published aligned cloud: %zu points (orange)", rgb_aligned->points.size());
                }

                // RViz 提示
                if (!rviz_guide_shown) {
                    RCLCPP_INFO(logger, "\n========== PC2R 处理完成 ==========");
                    RCLCPP_INFO(logger, "路径点数: %zu", Rviz_TrajectoryPose.size());
                    RCLCPP_INFO(logger, "\n========== RViz2 可视化指南 ==========");
                    RCLCPP_INFO(logger, "1. Set Fixed Frame to: %s", g_config.viz.fixed_frame.c_str());
                    RCLCPP_INFO(logger, "2. Add PointCloud2 -> /processing_pointcloud (查看所有点云)");
                    RCLCPP_INFO(logger, "3. Add Pose -> /Welding_Trajectory (红色箭头)");
                    RCLCPP_INFO(logger, "4. Add TF -> /tf (查看坐标系)");
                    RCLCPP_INFO(logger, "=====================================");
                    rviz_guide_shown = true;
                }

                rclcpp::spin_some(node);
                naptime.sleep();
                continue;
            }
            // ---------------------- 采集多视角点云 ----------------------
            if (collecting) {
                pcl::PointXYZ realsense_pos = read_realtime_pointcloud_frame(
                    g_config.data_path.dataset_folder, pointcloud_frameNum,
                    receive_capture_count, process_frame_count,
                    trajectoryPlanning_flag, cloud_ptr);

                if (cloud_ptr && !cloud_ptr->points.empty()) {
                    Cloud::Ptr cloud_xyz(new Cloud);
                    for (const auto& p : cloud_ptr->points) {
                        pcl::PointXYZ xyz;
                        xyz.x = p.x;
                        xyz.y = p.y;
                        xyz.z = p.z;
                        cloud_xyz->push_back(xyz);
                    }
                    multi_view_buffer.push_back(cloud_xyz);

                    Eigen::Vector3f pose(realsense_pos.x, realsense_pos.y, realsense_pos.z);
                    pose_buffer.push_back(pose);
                    receive_capture_count++;

                    RCLCPP_INFO(logger, "已采集: %zu 帧", multi_view_buffer.size());

                    if ((int)multi_view_buffer.size() >= max_frames) {
                        collecting = false;
                        processing_done = true;

                        Cloud::Ptr model_xyz(new Cloud);
                        for (const auto& p : model_pointcloud_display->points) {
                            pcl::PointXYZ xyz;
                            xyz.x = p.x;
                            xyz.y = p.y;
                            xyz.z = p.z;
                            model_xyz->push_back(xyz);
                        }

                        RCLCPP_INFO(logger, "开始 PC2R 处理...");
                        // 使用返回结构体
                        PC2RResult pc2r_result = trajectory_6DOF_generation_pc2r(
                            multi_view_buffer, pose_buffer,
                            model_xyz,
                            trajectoryPlanning_flag,
                            receive_capture_count, pub_pointcloud,
                            pointcloud_publisher, Welding_Trajectory_publisher,
                            naptime, logger);
                        
                        Rviz_TrajectoryPose = pc2r_result.trajectory;
                        processing_cloud = pc2r_result.edge_points;  // 获取焊缝边界点云
                    }
                }
            }
        }
        // ===================== 原始算法模式 =====================
        else {
            Rviz_TrajectoryPose = trajectory_6DOF_generation(
                read_realtime_pointcloud_frame(g_config.data_path.dataset_folder, pointcloud_frameNum,
                                            receive_capture_count, process_frame_count,
                                            trajectoryPlanning_flag, cloud_ptr),
                cloud_ptr, cloud_ptr_modelSeam,
                trajectoryPlanning_flag, receive_capture_count,
                pub_pointcloud, pointcloud_publisher, Welding_Trajectory_publisher,
                naptime);

            // 发布 TF
            for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++) {
                string markerFrame = Waypoint_markerName_creation(i);
                tf2::Transform waypoint_transform = create_waypoint_transform(i, Rviz_TrajectoryPose[i]);

                geometry_msgs::msg::TransformStamped waypoint_Marker;
                waypoint_Marker.header.stamp = node->get_clock()->now();
                waypoint_Marker.header.frame_id = g_config.viz.fixed_frame;
                waypoint_Marker.child_frame_id = markerFrame;
                waypoint_Marker.transform.translation.x = waypoint_transform.getOrigin().x();
                waypoint_Marker.transform.translation.y = waypoint_transform.getOrigin().y();
                waypoint_Marker.transform.translation.z = waypoint_transform.getOrigin().z();
                waypoint_Marker.transform.rotation.x = waypoint_transform.getRotation().x();
                waypoint_Marker.transform.rotation.y = waypoint_transform.getRotation().y();
                waypoint_Marker.transform.rotation.z = waypoint_transform.getRotation().z();
                waypoint_Marker.transform.rotation.w = waypoint_transform.getRotation().w();
                tf_broadcaster->sendTransform(waypoint_Marker);
            }

            // 发布模型点云
            publish_pointcloud_Rviz(g_config.viz.fixed_frame, model_pointcloud_display,
                                    pub_model_pointcloud_display, model_pointcloud_display_publisher);

            // 发布 Marker
            for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++) {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = g_config.viz.fixed_frame;
                marker.header.stamp = node->get_clock()->now();
                marker.ns = "waypoint_marker";
                marker.id = i;
                marker.type = visualization_msgs::msg::Marker::ARROW;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.points.resize(2);
                marker.points[0].x = Rviz_TrajectoryPose[i].position.x;
                marker.points[0].y = Rviz_TrajectoryPose[i].position.y;
                marker.points[0].z = Rviz_TrajectoryPose[i].position.z;
                marker.points[1].x = marker.points[0].x + Rviz_TrajectoryPose[i].orientation.x * g_config.viz.marker_arrow_length;
                marker.points[1].y = marker.points[0].y + Rviz_TrajectoryPose[i].orientation.y * g_config.viz.marker_arrow_length;
                marker.points[1].z = marker.points[0].z + Rviz_TrajectoryPose[i].orientation.z * g_config.viz.marker_arrow_length;
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 1.0;
                marker.scale.x = 0.008;
                marker.scale.y = 0.015;
                marker.scale.z = 0.04;
                vis_pub->publish(marker);
            }
            for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++) {
                visualization_msgs::msg::Marker sphere_marker;
                sphere_marker.header.frame_id = g_config.viz.fixed_frame;
                sphere_marker.header.stamp = node->get_clock()->now();
                sphere_marker.ns = "waypoint_sphere";
                sphere_marker.id = i;
                sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
                sphere_marker.action = visualization_msgs::msg::Marker::ADD;
                sphere_marker.pose.position.x = Rviz_TrajectoryPose[i].position.x;
                sphere_marker.pose.position.y = Rviz_TrajectoryPose[i].position.y;
                sphere_marker.pose.position.z = Rviz_TrajectoryPose[i].position.z;
                sphere_marker.pose.orientation.w = 1.0;
                sphere_marker.color.r = 0.0;
                sphere_marker.color.g = 1.0;
                sphere_marker.color.b = 0.0;
                sphere_marker.color.a = 1.0;
                sphere_marker.scale.x = g_config.viz.marker_sphere_size;
                sphere_marker.scale.y = g_config.viz.marker_sphere_size;
                sphere_marker.scale.z = g_config.viz.marker_sphere_size;
                vis_pub->publish(sphere_marker);
            }
        }

        rclcpp::spin_some(node);
        naptime.sleep();
    }

    rclcpp::shutdown();
    return 0;
}


// ============================================
// 回调函数
// ============================================

void color_Callback(const sensor_msgs::msg::Image::SharedPtr color_msg)
{
    try
    {
        color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("trajectory_planning"), 
                     "Could not convert from '%s' to 'bgr8'.", color_msg->encoding.c_str());
    }
    color_pic = color_ptr->image;
    waitKey(1);
}

void depth_Callback(const sensor_msgs::msg::Image::SharedPtr depth_msg)
{
    try
    {
        depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("trajectory_planning"), 
                     "Could not convert from '%s' to 'mono16'.", depth_msg->encoding.c_str());
    }
    depth_pic = depth_ptr->image;
    waitKey(1);
}

void pointcloud_storageFolder_Callback(const std_msgs::msg::String::SharedPtr msg)
{
    RCLCPP_INFO(rclcpp::get_logger("trajectory_planning"), "I heard the pointcloud_storageFolder");
    g_config.data_path.dataset_folder = msg->data.c_str();
    cout << "dataset_folder_path: " << g_config.data_path.dataset_folder << endl;
    receive_capture_count++;
}


// ============================================
// publish_pointcloud_Rviz函数
// ============================================
void publish_pointcloud_Rviz(string coordinate, PointCloud::Ptr pointloud,
                             sensor_msgs::msg::PointCloud2 pub_pointcloud,
                             rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    pcl::toROSMsg(*pointloud, pub_pointcloud);
    pub_pointcloud.header.frame_id = coordinate;
    pub_pointcloud.header.stamp = rclcpp::Clock().now();
    pointcloud_publisher->publish(pub_pointcloud);
}

// ============================================
// PC2R算法轨迹生成函数
// ============================================
// main.cpp - 修改 PC2RResult 函数

PC2RResult trajectory_6DOF_generation_pc2r(
    const std::vector<Cloud::Ptr>& multi_view_clouds,
    const std::vector<Eigen::Vector3f>& camera_poses,
    Cloud::Ptr cloud_ptr_modelSeam,
    bool& trajectoryPlanning_flag,
    int& receive_capture_count,
    sensor_msgs::msg::PointCloud2 pub_pointcloud,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher,
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime,
    rclcpp::Logger logger) {
    
    (void)trajectoryPlanning_flag;
    (void)receive_capture_count;
    (void)pub_pointcloud;
    (void)naptime;
    
    PC2RResult result;
    
    // 创建PC2R生成器并使用配置初始化
    PC2RWeldingGenerator pc2r_generator;
    configurePC2RGenerator(pc2r_generator);
    
    // 生成焊接路径
    std::vector<WeldingPathPoint> welding_path = pc2r_generator.generateWeldingPath(
        multi_view_clouds, camera_poses, logger, pointcloud_publisher);
    
    // ✅ 获取所有中间结果
    result.edge_points = pc2r_generator.getWeldSeamPoints();
    result.path_points = pc2r_generator.getPathPoints();
    result.stitched_cloud = pc2r_generator.getStitchedCloud();
    result.wsr_cloud = pc2r_generator.getWSRCloud();
    result.aligned_cloud = pc2r_generator.getAlignedCloud();
    
    RCLCPP_INFO(logger, "========== PC2R 处理结果 ==========");
    RCLCPP_INFO(logger, "边缘点: %zu 点", result.edge_points ? result.edge_points->points.size() : 0);
    RCLCPP_INFO(logger, "路径点: %zu 点", result.path_points ? result.path_points->points.size() : 0);
    RCLCPP_INFO(logger, "拼接点云: %zu 点", result.stitched_cloud ? result.stitched_cloud->points.size() : 0);
    RCLCPP_INFO(logger, "WSR点云: %zu 点", result.wsr_cloud ? result.wsr_cloud->points.size() : 0);
    RCLCPP_INFO(logger, "配准点云: %zu 点", result.aligned_cloud ? result.aligned_cloud->points.size() : 0);
    RCLCPP_INFO(logger, "焊接路径点: %zu 点", welding_path.size());
    RCLCPP_INFO(logger, "==================================");
    
    if (welding_path.empty()) {
        RCLCPP_ERROR(logger, "PC2R algorithm failed to generate welding path!");
        return result;
    }
    
    // 转换为ROS Pose消息
    for (const auto& point : welding_path) {
        geometry_msgs::msg::Pose pose = weldingPathPointToPose(point);
        result.trajectory.push_back(pose);
        Welding_Trajectory_publisher->publish(pose);
        rclcpp::sleep_for(std::chrono::milliseconds(10));
    }
    
    RCLCPP_INFO(logger, "PC2R algorithm generated %zu waypoints", result.trajectory.size());
    
    return result;
}
