// ============================================
// motion_planning.cpp - ROS2版本
// 修改点：ROS1头文件替换为ROS2版本，ros::ok() -> rclcpp::ok()
// ============================================

#include <motion_planning.hpp>

#include <rclcpp/rclcpp.hpp>
#include <math.h>
#include <iostream>   
#include <vector>

#include <running_flow.hpp>
#include <seam_location.hpp>
#include <motion_planning.hpp>
#include <transformation.hpp>

#include <std_msgs/msg/string.hpp>  // 修改：ROS2消息头文件
#include <dirent.h>

#include <image_transport/image_transport.hpp>  // 修改：添加.hpp
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>  // 修改：添加.hpp
#include <sensor_msgs/msg/image.hpp>  // 修改：ROS2消息类型
#include <sensor_msgs/msg/point_cloud2.hpp>  // 修改：ROS2消息类型
#include <geometry_msgs/msg/pose_stamped.hpp>  // 修改：ROS2消息类型
#include <geometry_msgs/msg/pose.hpp>  // 修改：ROS2消息类型

#include <tf2_ros/transform_listener.h>  // 修改：tf -> tf2
#include <tf2_ros/transform_broadcaster.h>  // 修改：tf -> tf2
#include <std_msgs/msg/bool.hpp>  // 修改：ROS2消息类型
#include <visualization_msgs/msg/marker.hpp>  // 修改：ROS2消息类型

// PCL lib
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <transformation.hpp>  // 修改：.h -> .hpp
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// 定义点云类型
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud; 
typedef pcl::PointCloud<pcl::PointXYZRGBL> PointCloudL;  
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::Normal> Normal;

using namespace cv;
using namespace std;
using namespace Eigen;

// ============================================
// 以下为原始算法代码，只修改了 ros::ok() -> rclcpp::ok()
// ============================================

//求所有的点距离之和：
vector<float> compute_Points_disSum(Cloud::Ptr cloud_ptr)
{
    vector<float> Point_DisSum;
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    { 
        float point_sum = 0;
        for (size_t j = 0; j < cloud_ptr->points.size(); j++)
        { 
            point_sum += Distance_two_Points(cloud_ptr->points[i], cloud_ptr->points[j]);
        }
        Point_DisSum.push_back(point_sum);
    }
    return Point_DisSum;
}

float Point_DisSum_min_Compute(vector<float> Point_DisSum)
{
    float Point_DisSum_min = 0;
    size_t Point_DisSum_min_index = 0;
    for (size_t j = 0; j < Point_DisSum.size(); j++)
    { 
        if (j == 0)
        {
            Point_DisSum_min = Point_DisSum[0];
        }
        if (Point_DisSum_min > Point_DisSum[j])
        {
            Point_DisSum_min = Point_DisSum[j];
            Point_DisSum_min_index = j;
        }
    }
    return Point_DisSum_min_index;
}

vector<float> cloud_GeometryCenter_pack(float Point_DisSum_min_index, Cloud::Ptr cloud_ptr)
{
    vector<float> GeometryCenter_pack;
    GeometryCenter_pack.push_back(Point_DisSum_min_index);
    GeometryCenter_pack.push_back(cloud_ptr->points[Point_DisSum_min_index].x);
    GeometryCenter_pack.push_back(cloud_ptr->points[Point_DisSum_min_index].y);
    GeometryCenter_pack.push_back(cloud_ptr->points[Point_DisSum_min_index].z);
    return GeometryCenter_pack;
}

vector<float> Compute_Segment_GeometryCenter(Cloud::Ptr cloud_ptr)
{
    vector<float> Point_DisSum = compute_Points_disSum(cloud_ptr);
    float Point_DisSum_min_index = Point_DisSum_min_Compute(Point_DisSum);
    return cloud_GeometryCenter_pack(Point_DisSum_min_index, cloud_ptr);
}

float Distance_two_Points(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    return sqrt(pow(p1.x - p2.x, 2) +
                pow(p1.y - p2.y, 2) +
                pow(p1.z - p2.z, 2));
}

vector<pcl::PointXYZ> Points_Exchange(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    pcl::PointXYZ p_temp;
    p_temp.x = p1.x; p_temp.y = p1.y; p_temp.z = p1.z;
    p1.x = p2.x; p1.y = p2.y; p1.z = p2.z;
    p2.x = p_temp.x; p2.y = p_temp.y; p2.z = p_temp.z;
    vector<pcl::PointXYZ> p;
    p.push_back(p1);
    p.push_back(p2);
    return p;
}

Cloud::Ptr Create_SeamCloud(Cloud::Ptr cloud_ptr)
{
    Cloud::Ptr seam_cloud(new Cloud);
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    { 
        pcl::PointXYZ ps;
        ps.x = cloud_ptr->points[i].x;
        ps.y = cloud_ptr->points[i].y;
        ps.z = cloud_ptr->points[i].z;
        seam_cloud->points.push_back(ps); 
    }
    return seam_cloud;
}

Cloud::Ptr Output_Boundary_SeamCloud(Cloud::Ptr seam_cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Boundary> boundaries;
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
    normEst.setInputCloud(seam_cloud);
    normEst.setSearchMethod(tree);
    normEst.setRadiusSearch(0.005);
    normEst.compute(*normals);
    cout << "normal size is " << normals->size() << endl;

    est.setInputCloud(seam_cloud);
    est.setInputNormals(normals);
    est.setSearchMethod(tree);
    est.setRadiusSearch(0.01);
    est.compute(boundaries);

    Cloud::Ptr boundPoints(new Cloud);
    int countBoundaries = 0;
    for (size_t i = 0; i < seam_cloud->size(); i++)
    {
        uint8_t x = (boundaries.points[i].boundary_point);
        int a = static_cast<int>(x);
        if (a)
        {
            (*boundPoints).push_back(seam_cloud->points[i]);
            countBoundaries++;
        }
    }    
    cout << "boudary size is：" << countBoundaries << endl;
    return boundPoints;
}

Cloud::Ptr Delete_noiseBoundary(Cloud::Ptr boundPoints)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ec_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ec_tree->setInputCloud(boundPoints);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> EC;
    EC.setClusterTolerance(0.002);
    EC.setMinClusterSize(1);
    EC.setMaxClusterSize(10000000);
    EC.setSearchMethod(ec_tree);
    EC.setInputCloud(boundPoints);
    EC.extract(cluster_indices);

    cout << "cluster_indices.size()" << cluster_indices.size() << endl;

    Cloud::Ptr seam_edge(new Cloud);
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            seam_edge->points.push_back(boundPoints->points[*pit]);  
        }
        break;
    }
    return seam_edge;
}

void cloud_ptr_show_creation(Cloud::Ptr seam_edge, PointCloud::Ptr cloud_ptr_show)
{
    for (size_t i = 0; i < seam_edge->points.size(); i++)
    { 
        pcl::PointXYZRGB p;
        p.x = seam_edge->points[i].x;
        p.y = seam_edge->points[i].y;
        p.z = seam_edge->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p); 
    }
}

vector<int> FindAllIndex_Around_OnePoint(Cloud::Ptr seam_edge, float i, float radius)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(seam_edge);
    vector<int> pointIdxRadiusSearch;
    vector<float> pointRadiusSquaredDistance;
    kdtree.radiusSearch(seam_edge->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);  
    return pointIdxRadiusSearch;
}

vector<Point3f> Distance_and_Index(Cloud::Ptr seam_edge, vector<int> pointIdxRadiusSearch)
{
    vector<Point3f> Distance_Points;
    for (size_t i = 0; i < pointIdxRadiusSearch.size(); i++)
    {         
        for (size_t j = i + 1; j < pointIdxRadiusSearch.size(); j++)
        { 
            float dis = Distance_two_Points(seam_edge->points[pointIdxRadiusSearch[i]], 
                                             seam_edge->points[pointIdxRadiusSearch[j]]);
            Point3f p; p.x = dis; p.y = i; p.z = j;
            Distance_Points.push_back(p);
        }
    }
    return Distance_Points;
}

float Distance_Points_max_index_Compute(vector<Point3f> Distance_Points)
{
    float Distance_Points_max = 0;
    size_t Distance_Points_max_index = 0;
    for (size_t j = 0; j < Distance_Points.size(); j++)
    { 
        if (j == 0)
        {
            Distance_Points_max = Distance_Points[0].x;
        }
        if (Distance_Points_max < Distance_Points[j].x)
        {
            Distance_Points_max = Distance_Points[j].x;
            Distance_Points_max_index = j;
        }
    }
    return Distance_Points_max_index;
}

Point3f Compute_Vector_TwoPoints(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    Point3f vector;
    vector.x = p1.x - p2.x;
    vector.y = p1.y - p2.y;
    vector.z = p1.z - p2.z;
    return vector;
}

float Compute_Included_Angle(Point3f vector1, Point3f vector2)
{
    float a_b = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;
    float a = sqrt(pow(vector1.x, 2) + pow(vector1.y, 2) + pow(vector1.z, 2));
    float b = sqrt(pow(vector2.x, 2) + pow(vector2.y, 2) + pow(vector2.z, 2));
    float theta = acos(a_b / (a * b)) * 180 / M_PI;
    return theta;
}

vector<float> Find_relevantPoint_onTheOherCurve(vector<Point3f> Distance_Points, 
                                                float Distance_Points_max_index, 
                                                Cloud::Ptr seam_edge, 
                                                vector<int> pointIdxRadiusSearch)
{
    vector<float> right_point;
    float index_1 = Distance_Points[Distance_Points_max_index].y;
    float index_2 = Distance_Points[Distance_Points_max_index].z;
    Point3f Tangency_vector = Compute_Vector_TwoPoints(seam_edge->points[pointIdxRadiusSearch[index_1]], 
                                                        seam_edge->points[pointIdxRadiusSearch[index_2]]);
    float find_right_point_flag = 0;
    float right_point_index = 0;
    float angle_threshold = 5;
    for (size_t i = 1; i < pointIdxRadiusSearch.size(); i++)
    {     
        Point3f each_vector = Compute_Vector_TwoPoints(seam_edge->points[pointIdxRadiusSearch[0]], 
                                                        seam_edge->points[pointIdxRadiusSearch[i]]);
        float theta = Compute_Included_Angle(Tangency_vector, each_vector);
        if (abs(abs(theta) - 90) <= angle_threshold)
        {
            find_right_point_flag = 1;
            right_point_index = i;
            break;
        }
    }
    right_point.push_back(find_right_point_flag);
    right_point.push_back(right_point_index);
    return right_point;
}

pcl::PointXYZ Compute_Single_PathPoint(Cloud::Ptr seam_edge, vector<int> pointIdxRadiusSearch, float right_point_index)
{
    pcl::PointXYZ path_point;
    path_point.x = (seam_edge->points[pointIdxRadiusSearch[0]].x + seam_edge->points[pointIdxRadiusSearch[right_point_index]].x) / 2;
    path_point.y = (seam_edge->points[pointIdxRadiusSearch[0]].y + seam_edge->points[pointIdxRadiusSearch[right_point_index]].y) / 2;
    path_point.z = (seam_edge->points[pointIdxRadiusSearch[0]].z + seam_edge->points[pointIdxRadiusSearch[right_point_index]].z) / 2;
    return path_point;
}

// ============================================
// 修改点：ros::ok() -> rclcpp::ok()
// ============================================
void DownSample_DeleteNoisePoint(Cloud::Ptr Path_Cloud, float radius)
{
    Cloud::Ptr Path_Point_Cloud(new Cloud);
    int loop_count = 0, loop_max = 50;
    while (rclcpp::ok())  // 修改：ros::ok() -> rclcpp::ok()
    {
        loop_count++;
        cout << "loop_count: " << loop_count << endl;
        for (size_t i = 0; i < Path_Cloud->points.size(); i++)
        { 
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            kdtree.setInputCloud(Path_Cloud);
            vector<int> pointIdxRadiusSearch;
            vector<float> pointRadiusSquaredDistance;
            kdtree.radiusSearch(Path_Cloud->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);  
            if (pointIdxRadiusSearch.size() <= 2)
            {
                continue;
            }
            pcl::PointXYZ m_p;
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
            { 
                pcl::PointXYZ p = Path_Cloud->points[pointIdxRadiusSearch[j]];
                m_p.x += p.x; m_p.y += p.y; m_p.z += p.z;
            }
            m_p.x = m_p.x / pointIdxRadiusSearch.size();
            m_p.y = m_p.y / pointIdxRadiusSearch.size();
            m_p.z = m_p.z / pointIdxRadiusSearch.size();
            Path_Point_Cloud->points.push_back(m_p);       
        }
        Path_Cloud->clear();
        for (size_t i = 0; i < Path_Point_Cloud->points.size(); i++)
        {  
            pcl::PointXYZ p = Path_Point_Cloud->points[i];
            Path_Cloud->points.push_back(p);       
        }
        Path_Point_Cloud->clear();
        if (loop_count == loop_max)
        {
            break;
        }
    }
}

Cloud::Ptr Merge_NearPoints(Cloud::Ptr Path_Cloud, float radius)
{
    Cloud::Ptr Path_Cloud_filtered(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(radius, radius, radius);
    voxel.setInputCloud(Path_Cloud);
    voxel.filter(*Path_Cloud_filtered);
    return Path_Cloud_filtered;
}

float Included_Value_TwoPoints(Point3f vector1, Point3f vector2)
{
    float a_b = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z;
    float a = sqrt(pow(vector1.x, 2) + pow(vector1.y, 2) + pow(vector1.z, 2));
    float b = sqrt(pow(vector2.x, 2) + pow(vector2.y, 2) + pow(vector2.z, 2));
    return (a_b / (a * b));
}

// ============================================
// 修改点：ros::ok() -> rclcpp::ok() (多处)
// ============================================
Cloud::Ptr Order_PathPoints_Cloud(Cloud::Ptr Path_Cloud_filtered, float radius)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(Path_Cloud_filtered);
    vector<int> pointIdxRadiusSearch;
    vector<float> pointRadiusSquaredDistance;

    vector<float> Path_Cloud_indexOrder; 
    vector<float> GeometryCenter = Compute_Segment_GeometryCenter(Path_Cloud_filtered);
    size_t nextPoint_index = GeometryCenter[0];
    size_t lastPoint_index = 0;
    Point3f vector_direction, last_vector_direction;
    vector<size_t> Path_Cloud_indexOrder_half1; 
    
    while (rclcpp::ok())  // 修改：ros::ok() -> rclcpp::ok()
    {
        int breakAll_flag = 0;
        if (Path_Cloud_indexOrder_half1.size() == 0)
        {
            kdtree.radiusSearch(Path_Cloud_filtered->points[nextPoint_index], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);  
            lastPoint_index = nextPoint_index;
            nextPoint_index = pointIdxRadiusSearch[1];
            vector_direction = Compute_Vector_TwoPoints(Path_Cloud_filtered->points[nextPoint_index], 
                                                        Path_Cloud_filtered->points[lastPoint_index]);
            Path_Cloud_indexOrder_half1.push_back(nextPoint_index); 
            cout << "half1.size(): " << Path_Cloud_indexOrder_half1.size() << endl;
            cout << "point: " << Path_Cloud_filtered->points[nextPoint_index] << endl << endl;
        }
        else
        {
            last_vector_direction.x = vector_direction.x; 
            last_vector_direction.y = vector_direction.y; 
            last_vector_direction.z = vector_direction.z;
            kdtree.radiusSearch(Path_Cloud_filtered->points[nextPoint_index], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);  
            lastPoint_index = nextPoint_index;
            for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)
            { 
                nextPoint_index = pointIdxRadiusSearch[j];
                vector_direction = Compute_Vector_TwoPoints(Path_Cloud_filtered->points[nextPoint_index], 
                                                            Path_Cloud_filtered->points[lastPoint_index]);
                float Included_Value = Included_Value_TwoPoints(last_vector_direction, vector_direction);
                if (Included_Value <= 1 && Included_Value >= 0)
                {
                    break;
                }
                if (j == pointIdxRadiusSearch.size() - 1 && Included_Value < 0)
                {
                    breakAll_flag = 1;
                }
            }
            if (breakAll_flag == 0)
            {
                Path_Cloud_indexOrder_half1.push_back(nextPoint_index); 
                cout << "half1.size(): " << Path_Cloud_indexOrder_half1.size() << endl;
                cout << "point: " << Path_Cloud_filtered->points[nextPoint_index] << endl << endl;
            }
        }
        if (breakAll_flag)
        {
            break;
        }
    }
 
    for (size_t j = 0; j < Path_Cloud_indexOrder_half1.size(); j++)
    {
        Path_Cloud_indexOrder.push_back(Path_Cloud_indexOrder_half1[Path_Cloud_indexOrder_half1.size() - 1 - j]);
    }

    vector<size_t> Path_Cloud_indexOrder_half2; 
    nextPoint_index = GeometryCenter[0];
    lastPoint_index = Path_Cloud_indexOrder_half1[0];
    
    while (rclcpp::ok())  // 修改：ros::ok() -> rclcpp::ok()
    {
        int breakAll_flag = 0;
        if (Path_Cloud_indexOrder_half2.size() == 0)
        {
            last_vector_direction = Compute_Vector_TwoPoints(Path_Cloud_filtered->points[nextPoint_index], 
                                                              Path_Cloud_filtered->points[lastPoint_index]);
            kdtree.radiusSearch(Path_Cloud_filtered->points[nextPoint_index], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);  
            lastPoint_index = nextPoint_index;
            for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)
            { 
                nextPoint_index = pointIdxRadiusSearch[j];
                vector_direction = Compute_Vector_TwoPoints(Path_Cloud_filtered->points[nextPoint_index], 
                                                            Path_Cloud_filtered->points[lastPoint_index]);
                float Included_Value = Included_Value_TwoPoints(last_vector_direction, vector_direction);
                if (Included_Value <= 1 && Included_Value >= 0)
                {
                    break;
                }
            }
            Path_Cloud_indexOrder_half2.push_back(nextPoint_index); 
            cout << "half2.size(): " << Path_Cloud_indexOrder_half2.size() << endl;
            cout << "point: " << Path_Cloud_filtered->points[nextPoint_index] << endl << endl;
        }
        else
        {
            last_vector_direction.x = vector_direction.x; 
            last_vector_direction.y = vector_direction.y; 
            last_vector_direction.z = vector_direction.z;
            kdtree.radiusSearch(Path_Cloud_filtered->points[nextPoint_index], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);  
            lastPoint_index = nextPoint_index;
            for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)
            { 
                nextPoint_index = pointIdxRadiusSearch[j];
                vector_direction = Compute_Vector_TwoPoints(Path_Cloud_filtered->points[nextPoint_index], 
                                                            Path_Cloud_filtered->points[lastPoint_index]);
                float Included_Value = Included_Value_TwoPoints(last_vector_direction, vector_direction);
                if (Included_Value <= 1 && Included_Value >= 0)
                {
                    break;
                }
                if (j == pointIdxRadiusSearch.size() - 1 && Included_Value < 0)
                {
                    breakAll_flag = 1;
                }
            }
            if (breakAll_flag == 0)
            {
                Path_Cloud_indexOrder_half2.push_back(nextPoint_index); 
                cout << "half2.size(): " << Path_Cloud_indexOrder_half2.size() << endl;
                cout << "point: " << Path_Cloud_filtered->points[nextPoint_index] << endl << endl;
            }
        }
        if (breakAll_flag)
        {
            break;
        }
    }

    Path_Cloud_indexOrder.push_back(GeometryCenter[0]);
    for (size_t j = 0; j < Path_Cloud_indexOrder_half2.size(); j++)
    {
        Path_Cloud_indexOrder.push_back(Path_Cloud_indexOrder_half2[j]);
    }

    Cloud::Ptr Path_Cloud_final(new Cloud);
    for (size_t j = 0; j < Path_Cloud_indexOrder.size(); j++)
    {
        pcl::PointXYZ p;
        p.x = Path_Cloud_filtered->points[Path_Cloud_indexOrder[j]].x;
        p.y = Path_Cloud_filtered->points[Path_Cloud_indexOrder[j]].y;
        p.z = Path_Cloud_filtered->points[Path_Cloud_indexOrder[j]].z;
        cout << j + 1 << " Path Points: " << p << endl;
        Path_Cloud_final->points.push_back(p);  
    }
    return Path_Cloud_final;
}

void Show_Ordered_PathPoints(Cloud::Ptr Path_Cloud_final, Cloud::Ptr cloud_ptr_origin, PointCloud::Ptr cloud_ptr_show)
{
    cloud_ptr_show->clear();
    for (size_t i = 0; i < cloud_ptr_origin->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = cloud_ptr_origin->points[i].x;
        p.y = cloud_ptr_origin->points[i].y;
        p.z = cloud_ptr_origin->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p);         
    }
    for (size_t j = 0; j < Path_Cloud_final->points.size(); j++)
    {
        pcl::PointXYZRGB p;
        p.x = Path_Cloud_final->points[j].x;
        p.y = Path_Cloud_final->points[j].y;
        p.z = Path_Cloud_final->points[j].z;
        p.b = 0;
        p.g = 0;
        p.r = 200;    
        cloud_ptr_show->points.push_back(p);         
    }
}

// ============================================
// 修改点：ros::ok() -> rclcpp::ok()
// ============================================
Cloud::Ptr Compute_All_PathPoints(Cloud::Ptr seam_edge)
{
    Cloud::Ptr Path_Cloud(new Cloud);
    cout << "seam_edge->points.size() " << seam_edge->points.size() << endl;

    for (size_t i = 0; i < seam_edge->points.size(); i++)
    { 
        float radius = 0.001;
        cout << "i-th: " << i << endl;
        while (rclcpp::ok())  // 修改：ros::ok() -> rclcpp::ok()
        {
            vector<int> pointIdxRadiusSearch = FindAllIndex_Around_OnePoint(seam_edge, i, radius);
            if (pointIdxRadiusSearch.size() <= 1)
            {
                radius += 0.001;
                continue;
            }
            if (radius >= 0.05)
            {
                break;  
            }

            vector<Point3f> Distance_Points = Distance_and_Index(seam_edge, pointIdxRadiusSearch);
            float Distance_Points_max_index = Distance_Points_max_index_Compute(Distance_Points);
            vector<float> right_point = Find_relevantPoint_onTheOherCurve(Distance_Points, Distance_Points_max_index, 
                                                                          seam_edge, pointIdxRadiusSearch);
            float find_right_point_flag = right_point[0];
            float right_point_index = right_point[1];

            if (find_right_point_flag == 0)
            {
                radius += 0.001;
                cout << "i-th: " << i << " " << "radius: " << radius << endl;
            }
            else
            {
                Path_Cloud->points.push_back(Compute_Single_PathPoint(seam_edge, pointIdxRadiusSearch, right_point_index)); 
                break;
            }
        }
    }
    return Path_Cloud;
}

void push_point_showCloud(Cloud::Ptr seam_edge, PointCloud::Ptr cloud_ptr_show)
{
    for (size_t i = 0; i < seam_edge->points.size(); i++)
    { 
        pcl::PointXYZRGB p;
        p.x = seam_edge->points[i].x;
        p.y = seam_edge->points[i].y;
        p.z = seam_edge->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p); 
    }
}

vector<Point3f> OriginWaypoint_torchDir_Unify(Cloud::Ptr PathPoint_Position, 
                                               vector<Point3f> Normal_Vector, 
                                               vector<Point3f> Cam_Position)
{
    vector<Point3f> Vector_point_to_camPosi;
    for (size_t i = 0; i < PathPoint_Position->points.size(); i++)
    {
        Point3f standard_vector;
        standard_vector.x = Cam_Position[i].x - PathPoint_Position->points[i].x;
        standard_vector.y = Cam_Position[i].y - PathPoint_Position->points[i].y;
        standard_vector.z = Cam_Position[i].z - PathPoint_Position->points[i].z;
        Vector_point_to_camPosi.push_back(standard_vector);
    }

    vector<float> Theta;
    for (size_t i = 0; i < Normal_Vector.size(); i++)
    {
        float a_b = Normal_Vector[i].x * Vector_point_to_camPosi[i].x +
                    Normal_Vector[i].y * Vector_point_to_camPosi[i].y +
                    Normal_Vector[i].z * Vector_point_to_camPosi[i].z;
        float a2 = sqrt(pow(Normal_Vector[i].x, 2) + pow(Normal_Vector[i].y, 2) + pow(Normal_Vector[i].z, 2));
        float b2 = sqrt(pow(Vector_point_to_camPosi[i].x, 2) + pow(Vector_point_to_camPosi[i].y, 2) + pow(Vector_point_to_camPosi[i].z, 2));
        float COS_ab = a_b / (a2 * b2);
        Theta.push_back(COS_ab);
    }

    for (size_t i = 0; i < Normal_Vector.size(); i++)
    {
        if (Theta[i] <= 0)
        {
            Normal_Vector[i] = Normal_Vector[i] * (-1);
        }
    }
    cout << "size:" << Normal_Vector.size() << endl << endl; 
    return Normal_Vector;
}

vector<Point3f> select_nearest_Cam_Position(vector<pcl::PointXYZ> all_realsense_position, Cloud::Ptr PathPoint_Position)
{
    vector<Point3f> All_Cam_Position;
    for (size_t i = 0; i < PathPoint_Position->points.size(); i++)
    {
        Point3f Cam_Position;
        vector<float> all_dis;
        for (size_t j = 0; j < all_realsense_position.size(); j++)
        {
            all_dis.push_back(Distance_two_Points(PathPoint_Position->points[i], all_realsense_position[j]));
        }
        float Distance_min = 0;
        size_t Distance_min_index = 0;
        for (size_t k = 0; k < all_dis.size(); k++)
        { 
            if (k == 0)
            {
                Distance_min = all_dis[0];
            }
            if (Distance_min > all_dis[k])
            {
                Distance_min = all_dis[k];
                Distance_min_index = k;
            }
        }
        Cam_Position.x = all_realsense_position[Distance_min_index].x;
        Cam_Position.y = all_realsense_position[Distance_min_index].y;
        Cam_Position.z = all_realsense_position[Distance_min_index].z;
        All_Cam_Position.push_back(Cam_Position);
    }
    return All_Cam_Position;
}

// ============================================
// 主要对外接口函数
// ============================================

Cloud::Ptr Extract_Seam_edge(Cloud::Ptr cloud_ptr, PointCloud::Ptr cloud_ptr_show)
{
    Cloud::Ptr seam_cloud = Create_SeamCloud(cloud_ptr);
    Cloud::Ptr boundPoints = Output_Boundary_SeamCloud(seam_cloud);
    push_point_showCloud(boundPoints, cloud_ptr_show);
    return boundPoints;
}

Cloud::Ptr PathPoint_Position_Generation(Cloud::Ptr seam_edge, Cloud::Ptr cloud_ptr_modelSeam, PointCloud::Ptr cloud_ptr_show)
{
    cloud_ptr_show_creation(seam_edge, cloud_ptr_show);
    Cloud::Ptr Path_Cloud = Compute_All_PathPoints(seam_edge);  
    cout << "Path_Cloud->points.size(): " << Path_Cloud->points.size() << endl;

    float radius = 0.003;
    DownSample_DeleteNoisePoint(Path_Cloud, radius);
    cout << "Path_Cloud->points.size(): " << Path_Cloud->points.size() << endl;

    Cloud::Ptr Path_Cloud_filtered = Merge_NearPoints(Path_Cloud, radius);
    cout << "Path_Cloud_filtered->points.size(): " << Path_Cloud_filtered->points.size() << endl;

    float radius_times = 10;
    Cloud::Ptr PathPoint_Position = Order_PathPoints_Cloud(Path_Cloud_filtered, radius * radius_times);
    cout << "PathPoint_Position->points.size(): " << PathPoint_Position->points.size() << endl;

    Show_Ordered_PathPoints(PathPoint_Position, cloud_ptr_modelSeam, cloud_ptr_show);
    return PathPoint_Position;
}

vector<Point3f> PathPoint_Orientation_Generation(Cloud::Ptr PathPoint_Position, Cloud::Ptr cloud_ptr, 
                                                  PointCloud::Ptr cloud_ptr_show, 
                                                  vector<pcl::PointXYZ> all_realsense_position)
{
    Cloud::Ptr all_cloud_ptr(new Cloud);
    for (size_t j = 0; j < PathPoint_Position->points.size(); j++)
    {
        pcl::PointXYZ p;
        p.x = PathPoint_Position->points[j].x;
        p.y = PathPoint_Position->points[j].y;
        p.z = PathPoint_Position->points[j].z;
        all_cloud_ptr->points.push_back(p);         
    }
    for (size_t j = 0; j < cloud_ptr->points.size(); j++)
    {
        pcl::PointXYZ p;
        p.x = cloud_ptr->points[j].x;
        p.y = cloud_ptr->points[j].y;
        p.z = cloud_ptr->points[j].z;
        all_cloud_ptr->points.push_back(p);         
    }

    float radius = 0.1;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(all_cloud_ptr);
    vector<int> pointIdxRadiusSearch;
    vector<float> pointRadiusSquaredDistance;

    vector<Point3f> Normal_Vector;
    for (size_t i = 0; i < all_cloud_ptr->points.size(); i++)
    {
        if (i < PathPoint_Position->points.size())
        {
            kdtree.radiusSearch(all_cloud_ptr->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
            
            Point3f Pm;
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
            {
                float xj = all_cloud_ptr->points[pointIdxRadiusSearch[j]].x;
                float yj = all_cloud_ptr->points[pointIdxRadiusSearch[j]].y;
                float zj = all_cloud_ptr->points[pointIdxRadiusSearch[j]].z;
                Pm.x += xj;
                Pm.y += yj;
                Pm.z += zj;
            }
            Pm.x = Pm.x / pointIdxRadiusSearch.size();
            Pm.y = Pm.y / pointIdxRadiusSearch.size();
            Pm.z = Pm.z / pointIdxRadiusSearch.size();

            vector<Point3f> V;
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
            { 
                float xj = all_cloud_ptr->points[pointIdxRadiusSearch[j]].x;
                float yj = all_cloud_ptr->points[pointIdxRadiusSearch[j]].y;
                float zj = all_cloud_ptr->points[pointIdxRadiusSearch[j]].z;
                Point3f v;
                v.x = Pm.x - xj;
                v.y = Pm.y - yj;
                v.z = Pm.z - zj;
                V.push_back(v);
            }

            Matrix3f yyT;
            yyT << 0, 0, 0, 0, 0, 0, 0, 0, 0;
            for (size_t j = 0; j < V.size(); j++)
            {
                yyT(0,0) += V[j].x * V[j].x; yyT(0,1) += V[j].x * V[j].y; yyT(0,2) += V[j].x * V[j].z;
                yyT(1,0) += V[j].y * V[j].x; yyT(1,1) += V[j].y * V[j].y; yyT(1,2) += V[j].y * V[j].z;
                yyT(2,0) += V[j].z * V[j].x; yyT(2,1) += V[j].z * V[j].y; yyT(2,2) += V[j].z * V[j].z;
            }

            JacobiSVD<Eigen::MatrixXf> svd(yyT, ComputeThinU | ComputeThinV);  
            Matrix3f U = svd.matrixU();   
            Point3f normal;
            normal.x = U(0, 2);
            normal.y = U(1, 2);
            normal.z = U(2, 2);
            Normal_Vector.push_back(normal);
        }
    }

    vector<Point3f> Cam_Position = select_nearest_Cam_Position(all_realsense_position, PathPoint_Position);
    vector<Point3f> Torch_Normal_Vector = OriginWaypoint_torchDir_Unify(PathPoint_Position, Normal_Vector, Cam_Position);
    cout << "Torch_Normal_Vector.size:" << Torch_Normal_Vector.size() << endl << endl;

    for (size_t i = 0; i < all_cloud_ptr->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = all_cloud_ptr->points[i].x;
        p.y = all_cloud_ptr->points[i].y;
        p.z = all_cloud_ptr->points[i].z;
        if (i < PathPoint_Position->points.size())
        {
            p.b = 0; p.g = 0; p.r = 200;
        }
        else
        {
            p.b = 200; p.g = 200; p.r = 200;
        }
        cloud_ptr_show->points.push_back(p);         
    }
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = cloud_ptr->points[i].x;
        p.y = cloud_ptr->points[i].y;
        p.z = cloud_ptr->points[i].z;
        p.b = 200; p.g = 200; p.r = 200;
        cloud_ptr_show->points.push_back(p);         
    }
    
    return Torch_Normal_Vector;
}
