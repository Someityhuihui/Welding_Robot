//焊接缝检测：
//1.删除大面积的平面点云
//2.提取边沿，两个面的连接处
//3.找到三条能构成等腰三角形的线，则判断这个区域属于焊接缝的一部分
//4.将所有可能的区域连起来，看能不能构成一条连续的空间曲线
//5.将所有检测出的焊接缝标号



// ============================================
// seam_location.hpp - ROS2版本
// 焊接缝检测模块头文件
// ============================================

#ifndef SEAM_LOCATION_HPP
#define SEAM_LOCATION_HPP

#include <rclcpp/rclcpp.hpp>
#include <math.h>
#include <iostream>   
#include <vector>
#include <ctime>
#include <Eigen/Dense>
#include <boost/thread/thread.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// PCL lib
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

// ============================================
// 类型定义（保持不变）
// ============================================
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud; 
typedef pcl::PointCloud<pcl::PointXYZRGBL> PointCloudL;  
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::Normal> Normal;

using namespace cv;
using namespace std;
using namespace Eigen;

// ============================================
// 函数声明（ROS2版本）
// ============================================

/**
 * @brief RGB图像焊缝提取（已注释，保留声明）
 */
void RGBimage_seam_extration(Mat color_pic, Mat depth_pic);

/**
 * @brief 读取点云文件并进行平滑处理
 * @param radius MLS平滑半径
 * @param cloud_ptr_show 用于显示的点云
 * @return 处理后的点云
 */
Cloud::Ptr read_pointcloud(float radius, PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 表面轮廓重建
 * @param radius 搜索半径
 * @param cloud_ptr 输入点云
 * @param cloud_ptr_show 输出显示点云
 */
void SurfaceProfile_Reconstruction(float radius, Cloud::Ptr cloud_ptr, PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 计算点云法向量
 * @param radius 搜索半径
 * @param cloud_ptr 输入点云
 * @param cloud_ptr_show 输出显示点云
 * @param Cam_Position 相机位置
 * @return 法向量列表
 */
vector<Point3f> PointNormal_Computation(float radius, Cloud::Ptr cloud_ptr, 
                                        PointCloud::Ptr cloud_ptr_show, 
                                        Point3f Cam_Position);

/**
 * @brief 删除平滑变化平面（提取焊缝候选区域）
 * @param radius 搜索半径
 * @param cloud_ptr 输入/输出点云
 * @param cloud_ptr_show 显示点云
 * @param Normal 法向量列表
 * @param pub_pointcloud 发布消息对象
 * @param pointcloud_publisher 点云发布器
 */
void Delete_SmoothChange_Plane(float radius, Cloud::Ptr cloud_ptr, 
                               PointCloud::Ptr cloud_ptr_show, 
                               vector<Point3f> Normal, 
                               sensor_msgs::msg::PointCloud2 pub_pointcloud, 
                               rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

/**
 * @brief 筛选候选焊缝（用户交互选择）
 * @param cloud_ptr 输入/输出点云
 * @param cloud_ptr_show 显示点云
 * @param pub_pointcloud 发布消息对象
 * @param pointcloud_publisher 点云发布器
 */
void Screen_Candidate_Seam(Cloud::Ptr cloud_ptr, PointCloud::Ptr cloud_ptr_show, 
                           sensor_msgs::msg::PointCloud2 pub_pointcloud, 
                           rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher);

/**
 * @brief 统一法向量方向（使所有法向量指向相机）
 * @param cloud_ptr 输入点云
 * @param cloud_ptr_show 显示点云
 * @param Normal 法向量列表
 * @param Cam_Position 相机位置
 * @return 统一方向后的法向量列表
 */
vector<Point3f> Pointnormal_Direction_Unify(Cloud::Ptr cloud_ptr, 
                                            PointCloud::Ptr cloud_ptr_show, 
                                            vector<Point3f> Normal, 
                                            Point3f Cam_Position);

#endif // SEAM_LOCATION_HPP