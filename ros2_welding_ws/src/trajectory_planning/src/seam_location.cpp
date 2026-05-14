// ============================================
// seam_location.cpp - ROS2版本
// 修改点：头文件替换为ROS2版本，ros::ok() -> rclcpp::ok()
// ============================================

#include <seam_location.hpp>

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

using namespace cv;
using namespace std;
using namespace Eigen;

// ============================================
// 辅助函数（无需修改）
// ============================================
void swap(int array[], int i, int j)
{
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

void BubbleSort1(int array[], int n)
{
    for (int i = 0; i < n-1; i++)
    {
        for (int j = i + 1; j < n-1; j++)
        {
            if (array[i] > array[j])
                swap(array, j, i);
        }
    }
}

// ============================================
// read_pointcloud函数（无ROS依赖，无需修改）
// ============================================
Cloud::Ptr read_pointcloud(float radius, PointCloud::Ptr cloud_ptr_show)
{
    Cloud::Ptr cloud_ptr(new Cloud);

    pcl::PCDReader reader;
    reader.read("/home/rick/Documents/a_system/src/trajectory_planning/src/driver_test_cloud.pcd", *cloud_ptr);

    cout << "PointCLoud size() " << cloud_ptr->width * cloud_ptr->height
         << " data points " << pcl::getFieldsList(*cloud_ptr) << "." << endl;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal> mls_points;
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setComputeNormals(true);
    mls.setInputCloud(cloud_ptr);
    mls.setPolynomialOrder(2);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(radius);
    mls.process(mls_points);
    cout << "smooth size(): " << mls_points.size() << endl << endl;

    Cloud::Ptr smooth_cloud(new Cloud);
    for (size_t i = 0; i < mls_points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = mls_points[i].x;
        p.y = mls_points[i].y;
        p.z = mls_points[i].z;
        smooth_cloud->points.push_back(p);
    }

    for (size_t i = 0; i < smooth_cloud->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = smooth_cloud->points[i].x;
        p.y = smooth_cloud->points[i].y + 0.1;
        p.z = smooth_cloud->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p);
    }

    cloud_ptr->points.clear();
    for (size_t i = 0; i < cloud_ptr_show->points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = cloud_ptr_show->points[i].x;
        p.y = cloud_ptr_show->points[i].y;
        p.z = cloud_ptr_show->points[i].z;
        cloud_ptr->points.push_back(p);
    }

    cout << "cloud_ptr_show->points.size()" << cloud_ptr->points.size() << endl << endl;
    return cloud_ptr;
}

// ============================================
// SurfaceProfile_Reconstruction函数
// ============================================
void SurfaceProfile_Reconstruction(float radius,
                                   Cloud::Ptr cloud_ptr,
                                   PointCloud::Ptr cloud_ptr_show)
{
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = cloud_ptr->points[i].x;
        p.y = cloud_ptr->points[i].y;
        p.z = cloud_ptr->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p);
    }
}

// ============================================
// Pointnormal_Direction_Unify函数声明
// ============================================
vector<Point3f> Pointnormal_Direction_Unify(Cloud::Ptr cloud_ptr,
                                            PointCloud::Ptr cloud_ptr_show,
                                            vector<Point3f> Normal,
                                            Point3f Cam_Position);

// ============================================
// PointNormal_Computation函数
// ============================================
vector<Point3f> PointNormal_Computation(float radius,
                                        Cloud::Ptr cloud_ptr,
                                        PointCloud::Ptr cloud_ptr_show,
                                        Point3f Cam_Position)
{
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = cloud_ptr->points[i].x;
        p.y = cloud_ptr->points[i].y;
        p.z = cloud_ptr->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_ptr);
    vector<int> pointIdxRadiusSearch;
    vector<float> pointRadiusSquaredDistance;

    vector<Point3f> Normal;
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        kdtree.radiusSearch(cloud_ptr->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        Point3f Pm;
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
        {
            float xj = cloud_ptr->points[pointIdxRadiusSearch[j]].x;
            float yj = cloud_ptr->points[pointIdxRadiusSearch[j]].y;
            float zj = cloud_ptr->points[pointIdxRadiusSearch[j]].z;
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
            float xj = cloud_ptr->points[pointIdxRadiusSearch[j]].x;
            float yj = cloud_ptr->points[pointIdxRadiusSearch[j]].y;
            float zj = cloud_ptr->points[pointIdxRadiusSearch[j]].z;
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
        Normal.push_back(normal);
    }

    return Pointnormal_Direction_Unify(cloud_ptr, cloud_ptr_show, Normal, Cam_Position);
}

// ============================================
// Pointnormal_Direction_Unify函数
// ============================================
vector<Point3f> Pointnormal_Direction_Unify(Cloud::Ptr cloud_ptr,
                                            PointCloud::Ptr cloud_ptr_show,
                                            vector<Point3f> Normal,
                                            Point3f Cam_Position)
{
    vector<Point3f> Vector_point_to_camPosi;
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        Point3f standard_vector;
        standard_vector.x = Cam_Position.x - cloud_ptr->points[i].x;
        standard_vector.y = Cam_Position.y - cloud_ptr->points[i].y;
        standard_vector.z = Cam_Position.z - cloud_ptr->points[i].z;
        Vector_point_to_camPosi.push_back(standard_vector);
    }

    vector<float> Theta;
    for (size_t i = 0; i < Normal.size(); i++)
    {
        float a_b = Normal[i].x * Vector_point_to_camPosi[i].x +
                    Normal[i].y * Vector_point_to_camPosi[i].y +
                    Normal[i].z * Vector_point_to_camPosi[i].z;
        float a2 = sqrt(pow(Normal[i].x, 2) + pow(Normal[i].y, 2) + pow(Normal[i].z, 2));
        float b2 = sqrt(pow(Vector_point_to_camPosi[i].x, 2) + pow(Vector_point_to_camPosi[i].y, 2) + pow(Vector_point_to_camPosi[i].z, 2));
        float COS_ab = a_b / (a2 * b2);
        Theta.push_back(COS_ab);
    }

    for (size_t i = 0; i < Normal.size(); i++)
    {
        if (Theta[i] <= 0)
        {
            Normal[i] = Normal[i] * (-1);
        }
    }

    cout << "size:" << Normal.size() << endl << endl;
    return Normal;
}

// ============================================
// Delete_SmoothChange_Plane函数
// 修改：ros::ok() -> rclcpp::ok()
// 修改：ros::Time::now() -> rclcpp::Clock().now()
// 修改：pointcloud_publisher.publish -> ->publish
// ============================================
void Delete_SmoothChange_Plane(float radius,
                              Cloud::Ptr cloud_ptr,
                              PointCloud::Ptr cloud_ptr_show,
                              vector<Point3f> Normal,
                              sensor_msgs::msg::PointCloud2 pub_pointcloud,
                              rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_ptr);
    vector<int> pointIdxRadiusSearch;
    vector<float> pointRadiusSquaredDistance;

    vector<float> PointVariance;
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        kdtree.radiusSearch(cloud_ptr->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        float ave_theta = 0;
        vector<float> PointTheta;
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
        {
            float a_b = Normal[pointIdxRadiusSearch[0]].x * Normal[pointIdxRadiusSearch[j]].x +
                        Normal[pointIdxRadiusSearch[0]].y * Normal[pointIdxRadiusSearch[j]].y +
                        Normal[pointIdxRadiusSearch[0]].z * Normal[pointIdxRadiusSearch[j]].z;
            ave_theta += a_b;
            PointTheta.push_back(a_b);
        }
        ave_theta = ave_theta / pointIdxRadiusSearch.size();

        float variance = 0;
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
        {
            variance += pow((PointTheta[j] - ave_theta), 2);
        }
        variance = variance / pointIdxRadiusSearch.size();
        PointVariance.push_back(variance);
    }
    cout << "size:" << PointVariance.size() << endl << endl;

    vector<float> PointDescriptor;
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        kdtree.radiusSearch(cloud_ptr->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        float ave_alpha = 0;
        vector<float> descriptor;
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
        {
            ave_alpha += PointVariance[pointIdxRadiusSearch[j]];
            descriptor.push_back(PointVariance[pointIdxRadiusSearch[j]]);
        }
        ave_alpha = ave_alpha / pointIdxRadiusSearch.size();

        float variance = 0;
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
        {
            variance += pow((descriptor[j] - ave_alpha), 2);
        }
        variance = variance / pointIdxRadiusSearch.size();
        PointDescriptor.push_back(variance);
    }
    cout << "size:" << PointDescriptor.size() << endl << endl;

    float PointDescriptor_max = 0, PointDescriptor_min = 0;
    for (size_t j = 0; j < PointDescriptor.size(); j++)
    {
        if (j == 0)
        {
            PointDescriptor_max = PointDescriptor[0];
            PointDescriptor_min = PointDescriptor[0];
        }
        if (PointDescriptor_max < PointDescriptor[j])
            PointDescriptor_max = PointDescriptor[j];
        if (PointDescriptor_min > PointDescriptor[j])
            PointDescriptor_min = PointDescriptor[j];
    }
    cout << "PointDescriptor_max: " << PointDescriptor_max << endl;
    cout << "PointDescriptor_min: " << PointDescriptor_min << endl;

    // 修改：ros::ok() -> rclcpp::ok()
    while (rclcpp::ok())
    {
        cout << "Please input screen_threshold: ";
        float screen_threshold = 0;
        cin >> screen_threshold;

        cloud_ptr_show->points.clear();
        for (size_t i = 0; i < cloud_ptr->points.size(); i++)
        {
            if (PointDescriptor[i] >= screen_threshold)
            {
                pcl::PointXYZRGB p;
                p.x = cloud_ptr->points[i].x;
                p.y = cloud_ptr->points[i].y;
                p.z = cloud_ptr->points[i].z;
                p.b = 200;
                p.g = 200;
                p.r = 200;
                cloud_ptr_show->points.push_back(p);
            }
        }
        cout << "cloud_ptr->points.size(): " << cloud_ptr->points.size() << endl;
        cout << "cloud_ptr_show->points.size(): " << cloud_ptr_show->points.size() << endl;

        pcl::toROSMsg(*cloud_ptr_show, pub_pointcloud);
        pub_pointcloud.header.frame_id = "base_link";
        pub_pointcloud.header.stamp = rclcpp::Clock().now();  // 修改：ros::Time::now()
        pointcloud_publisher->publish(pub_pointcloud);       // 修改：-> 操作符

        cout << endl << "Keep the pointcloud or not? yes or xxxx ";
        string keep_flag;
        cin >> keep_flag;
        if (keep_flag == "yes")
        {
            cloud_ptr->points.clear();
            for (size_t i = 0; i < cloud_ptr_show->points.size(); i++)
            {
                pcl::PointXYZ p;
                p.x = cloud_ptr_show->points[i].x;
                p.y = cloud_ptr_show->points[i].y;
                p.z = cloud_ptr_show->points[i].z;
                cloud_ptr->points.push_back(p);
            }
            break;
        }
        cloud_ptr_show->points.clear();
        cout << endl;
    }
}

// ============================================
// Screen_Candidate_Seam函数
// 修改：ros::ok() -> rclcpp::ok()
// 修改：ros::Time::now() -> rclcpp::Clock().now()
// 修改：pointcloud_publisher.publish -> ->publish
// ============================================
void Screen_Candidate_Seam(Cloud::Ptr cloud_ptr,
                          PointCloud::Ptr cloud_ptr_show,
                          sensor_msgs::msg::PointCloud2 pub_pointcloud,
                          rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ec_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ec_tree->setInputCloud(cloud_ptr);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> EC;
    EC.setClusterTolerance(0.002);
    EC.setMinClusterSize(500);
    EC.setMaxClusterSize(10000000);
    EC.setSearchMethod(ec_tree);
    EC.setInputCloud(cloud_ptr);
    EC.extract(cluster_indices);

    cout << "ec_tree_cloud->points.size(): " << cloud_ptr->points.size() << endl;
    cout << "cluster_indices.size(): " << cluster_indices.size() << endl;

    vector<vector<float>> seam_cluster_all;
    for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        vector<float> seam_cluster;
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            seam_cluster.push_back(*pit);
        }
        seam_cluster_all.push_back(seam_cluster);
    }

    int seam_label = 0;
    // 修改：ros::ok() -> rclcpp::ok()
    while (rclcpp::ok())
    {
        cout << "seam_cluster_all.size(): " << seam_cluster_all.size() << endl;
        cout << "Please input index of seam cluster 0-max : ";
        float index_seam_cluster = 0;
        cin >> index_seam_cluster;

        cloud_ptr_show->points.clear();
        for (size_t i = 0; i < seam_cluster_all[index_seam_cluster].size(); i++)
        {
            pcl::PointXYZRGB p;
            int idx = seam_cluster_all[index_seam_cluster][i];
            p.x = cloud_ptr->points[idx].x;
            p.y = cloud_ptr->points[idx].y;
            p.z = cloud_ptr->points[idx].z;
            p.b = 200;
            p.g = 200;
            p.r = 200;
            cloud_ptr_show->points.push_back(p);
        }

        pcl::toROSMsg(*cloud_ptr_show, pub_pointcloud);
        pub_pointcloud.header.frame_id = "base_link";
        pub_pointcloud.header.stamp = rclcpp::Clock().now();  // 修改：ros::Time::now()
        pointcloud_publisher->publish(pub_pointcloud);       // 修改：-> 操作符

        cout << endl << "Keep the pointcloud or not? yes or xxxx " << endl;
        string keep_flag;
        cin >> keep_flag;
        if (keep_flag == "yes")
        {
            seam_label = index_seam_cluster;
            break;
        }
        cloud_ptr_show->points.clear();
        cout << endl;
    }

    cloud_ptr->points.clear();
    for (size_t i = 0; i < seam_cluster_all[seam_label].size(); i++)
    {
        pcl::PointXYZ p;
        int idx = seam_cluster_all[seam_label][i];
        p.x = cloud_ptr->points[idx].x;
        p.y = cloud_ptr->points[idx].y;
        p.z = cloud_ptr->points[idx].z;
        cloud_ptr->points.push_back(p);
    }
}
