//运动规划：
//1.确定焊接方向
//2.分割焊接缝
//3.计算焊接点
//4.为每个焊接点计算旋转方向





// ============================================
// motion_planning.hpp - ROS2版本
// 焊接路径规划模块头文件
// ============================================

#ifndef MOTION_PLANNING_HPP
#define MOTION_PLANNING_HPP

#include <rclcpp/rclcpp.hpp>
#include <math.h>
#include <iostream>   
#include <vector>
#include <ctime>
#include <Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>

// PCL lib
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>

#include <pcl/features/boundary.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>

using namespace cv;
using namespace std;
using namespace Eigen;

// ============================================
// 类型定义
// ============================================
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud; 
typedef pcl::PointCloud<pcl::PointXYZRGBL> PointCloudL;  
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::Normal> Normal;

// ============================================
// 函数声明
// ============================================

/**
 * @brief 计算点云中所有点的距离和
 */
vector<float> compute_Points_disSum(Cloud::Ptr cloud_ptr);

/**
 * @brief 找到距离和最小的点索引
 */
float Point_DisSum_min_Compute(vector<float> Point_DisSum);

/**
 * @brief 将几何中心打包为向量
 */
vector<float> cloud_GeometryCenter_pack(float Point_DisSum_min_index, Cloud::Ptr cloud_ptr);

/**
 * @brief 计算点云段的几何中心
 */
vector<float> Compute_Segment_GeometryCenter(Cloud::Ptr cloud_ptr);

/**
 * @brief 计算两点之间的欧氏距离
 */
float Distance_two_Points(pcl::PointXYZ p1, pcl::PointXYZ p2);

/**
 * @brief 交换两个点的位置
 */
vector<pcl::PointXYZ> Points_Exchange(pcl::PointXYZ p1, pcl::PointXYZ p2);

/**
 * @brief 创建焊缝点云
 */
Cloud::Ptr Create_SeamCloud(Cloud::Ptr cloud_ptr);

/**
 * @brief 输出焊缝边界点云
 */
Cloud::Ptr Output_Boundary_SeamCloud(Cloud::Ptr seam_cloud);

/**
 * @brief 删除噪声边界点
 */
Cloud::Ptr Delete_noiseBoundary(Cloud::Ptr boundPoints);

/**
 * @brief 创建显示点云
 */
void cloud_ptr_show_creation(Cloud::Ptr seam_edge, PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 查找某点周围指定半径内的所有点索引
 */
vector<int> FindAllIndex_Around_OnePoint(Cloud::Ptr seam_edge, float i, float radius);

/**
 * @brief 计算点之间的距离和索引
 */
vector<Point3f> Distance_and_Index(Cloud::Ptr seam_edge, vector<int> pointIdxRadiusSearch);

/**
 * @brief 找到最大距离的索引
 */
float Distance_Points_max_index_Compute(vector<Point3f> Distance_Points);

/**
 * @brief 计算两点之间的向量
 */
Point3f Compute_Vector_TwoPoints(pcl::PointXYZ p1, pcl::PointXYZ p2);

/**
 * @brief 计算两个向量之间的夹角
 */
float Compute_Included_Angle(Point3f vector1, Point3f vector2);

/**
 * @brief 在另一条曲线上找到相关点
 */
vector<float> Find_relevantPoint_onTheOherCurve(vector<Point3f> Distance_Points, 
                                                float Distance_Points_max_index, 
                                                Cloud::Ptr seam_edge, 
                                                vector<int> pointIdxRadiusSearch);

/**
 * @brief 计算单个路径点
 */
pcl::PointXYZ Compute_Single_PathPoint(Cloud::Ptr seam_edge, 
                                        vector<int> pointIdxRadiusSearch, 
                                        float right_point_index);

/**
 * @brief 下采样并删除噪声点
 */
void DownSample_DeleteNoisePoint(Cloud::Ptr Path_Cloud, float radius);

/**
 * @brief 合并邻近点
 */
Cloud::Ptr Merge_NearPoints(Cloud::Ptr Path_Cloud, float radius);

/**
 * @brief 计算两点向量的点积值
 */
float Included_Value_TwoPoints(Point3f vector1, Point3f vector2);

/**
 * @brief 对路径点进行排序
 */
Cloud::Ptr Order_PathPoints_Cloud(Cloud::Ptr Path_Cloud_filtered, float radius);

/**
 * @brief 显示排序后的路径点
 */
void Show_Ordered_PathPoints(Cloud::Ptr Path_Cloud_final, 
                             Cloud::Ptr cloud_ptr_origin, 
                             PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 计算所有路径点
 */
Cloud::Ptr Compute_All_PathPoints(Cloud::Ptr seam_edge);

/**
 * @brief 推送点云到显示点云
 */
void push_point_showCloud(Cloud::Ptr seam_edge, PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 统一焊枪方向
 */
vector<Point3f> OriginWaypoint_torchDir_Unify(Cloud::Ptr PathPoint_Position, 
                                               vector<Point3f> Torch_Normal_Vector, 
                                               vector<Point3f> Cam_Position);

/**
 * @brief 选择最近的相机位置
 */
vector<Point3f> select_nearest_Cam_Position(vector<pcl::PointXYZ> all_realsense_position, 
                                             Cloud::Ptr PathPoint_Position);

/**
 * @brief 提取焊缝边缘
 */
Cloud::Ptr Extract_Seam_edge(Cloud::Ptr cloud_ptr, PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 生成路径点位置
 */
Cloud::Ptr PathPoint_Position_Generation(Cloud::Ptr seam_edge, 
                                          Cloud::Ptr cloud_ptr_origin, 
                                          PointCloud::Ptr cloud_ptr_show);

/**
 * @brief 生成路径点方向
 */
vector<Point3f> PathPoint_Orientation_Generation(Cloud::Ptr PathPoint_Position,
                                                  Cloud::Ptr cloud_ptr, 
                                                  PointCloud::Ptr cloud_ptr_show, 
                                                  vector<pcl::PointXYZ> all_realsense_position);

#endif // MOTION_PLANNING_HPP
