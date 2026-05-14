#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>


// ROS2 头文件
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core.hpp>

// 类型别名
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloud = pcl::PointCloud<pcl::PointXYZRGB>;
using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;

using namespace std;
// 添加 Point3f 类型定义（替代 OpenCV 的 Point3f）
struct Point3f {
    float x, y, z;
    Point3f() : x(0), y(0), z(0) {}
    Point3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};
using namespace Eigen;


// 右手坐标系
void rotate_z(float x, float y, float z, float angle, float* x_output, float* y_output, float* z_output) 
{
    float atopi = angle / 180.0 * M_PI;
    *x_output = x * cos(atopi) - y * sin(atopi);
    *y_output = y * cos(atopi) + x * sin(atopi);
    *z_output = z;
}

void rotate_x(float x, float y, float z, float angle, float* x_output, float* y_output, float* z_output)
{
    float atopi = angle / 180.0 * M_PI;
    *z_output = z * cos(atopi) + y * sin(atopi);
    *y_output = y * cos(atopi) - z * sin(atopi);
    *x_output = x;
}

void rotate_y(float x, float y, float z, float angle, float* x_output, float* y_output, float* z_output)
{
    float atopi = angle / 180.0 * M_PI;
    *x_output = x * cos(atopi) - z * sin(atopi);
    *z_output = z * cos(atopi) + x * sin(atopi);
    *y_output = y;
}

// 四元数 -> 旋转矩阵
void Quaternion_to_RotationMatrix(float x, float y, float z, float w, float R[9])
{
    R[0]  = 1 - 2 * y * y - 2 * z * z;
    R[1]  =     2 * x * y - 2 * z * w;
    R[2]  =     2 * x * z + 2 * y * w;

    R[3]  =     2 * x * y + 2 * z * w;
    R[4]  = 1 - 2 * x * x - 2 * z * z;
    R[5]  =     2 * y * z - 2 * x * w;

    R[6]  =     2 * x * z - 2 * y * w;
    R[7]  =     2 * y * z + 2 * x * w;
    R[8]  = 1 - 2 * x * x - 2 * y * y;
}

// 欧拉角 -> 四元数
void euler_to_quaternion(float Yaw, float Pitch, float Roll, float Q[4])
{
    float yaw   = Yaw   * M_PI / 180 ;
    float pitch = Roll  * M_PI / 180 ;
    float roll  = Pitch * M_PI / 180 ;

    float qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2);
    float qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2);
    float qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2);
    float qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2);
 
    Q[0] = qx;
    Q[1] = qy;
    Q[2] = qz;
    Q[3] = qw;
}

// 欧拉角 -> 旋转矩阵
void euler_to_RotationMatrix(float Yaw, float Pitch, float Roll, float R[9])
{
    float yaw   = Yaw   * M_PI / 180 ;
    float pitch = Roll  * M_PI / 180 ;
    float roll  = Pitch * M_PI / 180 ;

    float c1 = cos(yaw);
    float c2 = cos(pitch);
    float c3 = cos(roll);
    float s1 = sin(yaw);
    float s2 = sin(pitch);
    float s3 = sin(roll);

    R[0]  = c1 * c3 + s1 * s2 * s3;
    R[1]  = c3 * s1 * s2 - c1 * s3;
    R[2]  = c2 * s1;

    R[3]  = c2 * s3;
    R[4]  = c2 * c3;
    R[5]  = -s2;

    R[6]  = c1 * s2 * s3 - s1 * c3;
    R[7]  = s1 * s3 + c1 * c3 * s2;
    R[8]  = c1 * c2;
}

geometry_msgs::msg::Pose Torch_to_End_transform(const geometry_msgs::msg::TransformStamped& transform_tool02torch)
{
    geometry_msgs::msg::Pose res_pose;

    Eigen::Quaterniond q(
        transform_tool02torch.transform.rotation.w, 
        transform_tool02torch.transform.rotation.x, 
        transform_tool02torch.transform.rotation.y,
        transform_tool02torch.transform.rotation.z
    );

    Matrix3d RotationMatrix = q.toRotationMatrix();

    Matrix4d T_tool02torch;
    T_tool02torch << RotationMatrix(0,0), RotationMatrix(0,1), RotationMatrix(0,2), transform_tool02torch.transform.translation.x,
                     RotationMatrix(1,0), RotationMatrix(1,1), RotationMatrix(1,2), transform_tool02torch.transform.translation.y,
                     RotationMatrix(2,0), RotationMatrix(2,1), RotationMatrix(2,2), transform_tool02torch.transform.translation.z,
                     0                  , 0                  , 0                  , 1;

    return res_pose;
}

void Base_to_End_transform(int &receive_pose_flag, const geometry_msgs::msg::TransformStamped& transform)
{
    if (receive_pose_flag == 3)
    {
        geometry_msgs::msg::Pose Base_End;
        Base_End.position.x = transform.transform.translation.x;
        Base_End.position.y = transform.transform.translation.y;
        Base_End.position.z = transform.transform.translation.z;
        Base_End.orientation.x = transform.transform.rotation.x;
        Base_End.orientation.y = transform.transform.rotation.y;
        Base_End.orientation.z = transform.transform.rotation.z;
        Base_End.orientation.w = transform.transform.rotation.w;

        std::cout << "Base_End.position:" 
                  << Base_End.position.x << " " 
                  << Base_End.position.y << " " 
                  << Base_End.position.z << std::endl;
        receive_pose_flag = 0;
    }
}

Cloud::Ptr cloud_ptr_origin_copy(Cloud::Ptr cloud_ptr_new)
{
    Cloud::Ptr cloud_ptr_origin (new Cloud);
    for(std::size_t i = 0; i < cloud_ptr_new->points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = cloud_ptr_new->points[i].x;
        p.y = cloud_ptr_new->points[i].y;
        p.z = cloud_ptr_new->points[i].z;
        cloud_ptr_origin->points.push_back( p );    
    }
    return cloud_ptr_origin;
}

tf2::Transform Waypoint_markerTransform_creation(int i, const geometry_msgs::msg::Pose& P)
{
    tf2::Quaternion rotation (
        P.orientation.x,
        P.orientation.y, 
        P.orientation.z,
        P.orientation.w
    );

    tf2::Vector3 origin (
        P.position.x, 
        P.position.y,
        P.position.z
    );

    tf2::Transform t (rotation, origin);
    return t;
}

std::string Waypoint_markerName_creation( int i )
{
    std::string markerFrame = "waypoint_marker_";
    std::stringstream out;
    out << i;
    std::string id_string = out.str();
    markerFrame += id_string;
    return markerFrame;
}

void pathpoint_cut_head_tail(int points_cut_count, 
                             Cloud::Ptr PathPoint_Position, 
                             std::vector<Eigen::Vector3f> Torch_Normal_Vector, 
                             Cloud::Ptr PathPoint_Position_final, 
                             Cloud::Ptr Torch_Normal_Vector_final)
{
    for(int i = points_cut_count; i < (int)PathPoint_Position->points.size() - points_cut_count; i++)
    {
        pcl::PointXYZ p;
        p.x = PathPoint_Position->points[i].x;
        p.y = PathPoint_Position->points[i].y;
        p.z = PathPoint_Position->points[i].z;
        PathPoint_Position_final->points.push_back( p );    
    }
    for(int i = points_cut_count; i < (int)Torch_Normal_Vector.size() - points_cut_count; i++)
    {
        pcl::PointXYZ  vector;
        vector.x = Torch_Normal_Vector[i].x();
        vector.y = Torch_Normal_Vector[i].y();
        vector.z = Torch_Normal_Vector[i].z();
        Torch_Normal_Vector_final->points.push_back(vector);
    }
    std::cout << "PathPoint_Position_final->points.size(): " << PathPoint_Position_final->points.size()  << std::endl;
    std::cout << "Torch_Normal_Vector_final.size(): "        << Torch_Normal_Vector_final->points.size() << std::endl << std::endl;

    for(std::size_t i = 0; i < Torch_Normal_Vector_final->points.size(); i++)
    {
        std::cout << "Torch_Normal_Vector_final"        << Torch_Normal_Vector_final->points[i] << std::endl;
    }
}

void X_Normal_Vector_vertical(Cloud::Ptr X_Normal_Vector,
                              Cloud::Ptr PathPoint_Position_final, 
                              Cloud::Ptr Torch_Normal_Vector_final)
{
    for(std::size_t i = 0; i < PathPoint_Position_final->points.size()-1; i++)
    {
        pcl::PointXYZ temp_vector;

        temp_vector.x = PathPoint_Position_final->points[i+1].x - PathPoint_Position_final->points[i].x;
        temp_vector.y = PathPoint_Position_final->points[i+1].y - PathPoint_Position_final->points[i].y;
        temp_vector.z = PathPoint_Position_final->points[i+1].z - PathPoint_Position_final->points[i].z;

        Vector3d v(                  temp_vector.x,                   temp_vector.y,                   temp_vector.z);
        Vector3d w(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d u = v.cross(w);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ x_normal_vector;
        x_normal_vector.x = u[0] / n;
        x_normal_vector.y = u[1] / n;
        x_normal_vector.z = u[2] / n; 
        
        X_Normal_Vector->points.push_back(x_normal_vector);

        if(i == PathPoint_Position_final->points.size()-1-1)
        {
            X_Normal_Vector->points.push_back(x_normal_vector);
        }
    }
}

void Y_Z_Normal_Vector_vertical(Cloud::Ptr Y_Normal_Vector, 
                                Cloud::Ptr Z_Normal_Vector, 
                                Cloud::Ptr X_Normal_Vector, 
                                Cloud::Ptr Torch_Normal_Vector_final)
{
    for(std::size_t i = 0; i < X_Normal_Vector->points.size(); i++)
    {
        Vector3d v(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d w(           X_Normal_Vector->points[i].x,            X_Normal_Vector->points[i].y,            X_Normal_Vector->points[i].z);
        Vector3d u = v.cross(w);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ y_normal_vector;
        y_normal_vector.x = u[0] / n;
        y_normal_vector.y = u[1] / n;
        y_normal_vector.z = u[2] / n; 
        Y_Normal_Vector->points.push_back(y_normal_vector);

        pcl::PointXYZ z_normal_vector;
        z_normal_vector.x = -Torch_Normal_Vector_final->points[i].x;
        z_normal_vector.y = -Torch_Normal_Vector_final->points[i].y;
        z_normal_vector.z = -Torch_Normal_Vector_final->points[i].z;
        
        Z_Normal_Vector->points.push_back(z_normal_vector);
    }
}

void direction_modification_vertical( Cloud::Ptr X_Normal_Vector,
                                      Cloud::Ptr Y_Normal_Vector, 
                                      Cloud::Ptr Z_Normal_Vector, 
                                      Cloud::Ptr PathPoint_Position_final)
{
    X_Normal_Vector->points.clear();
    for(std::size_t i = 0; i < Y_Normal_Vector->points.size(); i++)
    {
        Vector3d k(0-PathPoint_Position_final->points[i].x,
                   0-PathPoint_Position_final->points[i].y,
                   0-PathPoint_Position_final->points[i].z);

        Vector3d h(Y_Normal_Vector->points[i].x,
                   Y_Normal_Vector->points[i].y,
                   Y_Normal_Vector->points[i].z);

        if(k.dot(h) < 0)
        {
            Y_Normal_Vector->points[i].x = -Y_Normal_Vector->points[i].x;
            Y_Normal_Vector->points[i].y = -Y_Normal_Vector->points[i].y;
            Y_Normal_Vector->points[i].z = -Y_Normal_Vector->points[i].z;
        }

        Vector3d v(Y_Normal_Vector->points[i].x, Y_Normal_Vector->points[i].y, Y_Normal_Vector->points[i].z);
        Vector3d w(Z_Normal_Vector->points[i].x, Z_Normal_Vector->points[i].y, Z_Normal_Vector->points[i].z);
        Vector3d u = v.cross(w);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ x_normal_vector;
        x_normal_vector.x = u[0] / n;
        x_normal_vector.y = u[1] / n;
        x_normal_vector.z = u[2] / n; 
        
        X_Normal_Vector->points.push_back(x_normal_vector);
    }
}

void Y_Normal_Vector_horizontal(Cloud::Ptr Y_Normal_Vector,
                                Cloud::Ptr PathPoint_Position_final, 
                                Cloud::Ptr Torch_Normal_Vector_final)
{
    for(std::size_t i = 0; i < PathPoint_Position_final->points.size()-1; i++)
    {
        pcl::PointXYZ temp_vector;

        temp_vector.x = PathPoint_Position_final->points[i+1].x - PathPoint_Position_final->points[i].x;
        temp_vector.y = PathPoint_Position_final->points[i+1].y - PathPoint_Position_final->points[i].y;
        temp_vector.z = PathPoint_Position_final->points[i+1].z - PathPoint_Position_final->points[i].z;

        Vector3d v(                  temp_vector.x,                   temp_vector.y,                   temp_vector.z);
        Vector3d w(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d u = w.cross(v);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ y_normal_vector;
        y_normal_vector.x = u[0] / n;
        y_normal_vector.y = u[1] / n;
        y_normal_vector.z = u[2] / n; 
        
        Y_Normal_Vector->points.push_back(y_normal_vector);

        if(i == PathPoint_Position_final->points.size()-1-1)
        {
            Y_Normal_Vector->points.push_back(y_normal_vector);
        }
    }
    std::cout << "Y_Normal_Vector: " << Y_Normal_Vector->points.size() << std::endl << std::endl;
}

void X_Z_Normal_Vector_horizontal(Cloud::Ptr Y_Normal_Vector, 
                                  Cloud::Ptr Z_Normal_Vector, 
                                  Cloud::Ptr X_Normal_Vector, 
                                  Cloud::Ptr Torch_Normal_Vector_final)
{
    for(std::size_t i = 0; i < Y_Normal_Vector->points.size(); i++)
    {
        Vector3d v(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d w(           Y_Normal_Vector->points[i].x,            Y_Normal_Vector->points[i].y,            Y_Normal_Vector->points[i].z);
        Vector3d u = w.cross(v);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ x_normal_vector;
        x_normal_vector.x = u[0] / n;
        x_normal_vector.y = u[1] / n;
        x_normal_vector.z = u[2] / n; 
        X_Normal_Vector->points.push_back(x_normal_vector);

        pcl::PointXYZ z_normal_vector;
        z_normal_vector.x = -Torch_Normal_Vector_final->points[i].x;
        z_normal_vector.y = -Torch_Normal_Vector_final->points[i].y;
        z_normal_vector.z = -Torch_Normal_Vector_final->points[i].z;
        
        Z_Normal_Vector->points.push_back(z_normal_vector);
    }
    std::cout << "X_Normal_Vector: " << X_Normal_Vector->points.size()  << std::endl;
    std::cout << "Z_Normal_Vector: " << Z_Normal_Vector->points.size()  << std::endl  << std::endl;
}

void direction_modification_horizontal( Cloud::Ptr X_Normal_Vector,
                                        Cloud::Ptr Y_Normal_Vector, 
                                        Cloud::Ptr Z_Normal_Vector, 
                                        Cloud::Ptr PathPoint_Position_final)
{
    X_Normal_Vector->points.clear();
    for(std::size_t i = 0; i < Y_Normal_Vector->points.size(); i++)
    {
        Vector3d k(0, -1, 0);

        Vector3d h(Y_Normal_Vector->points[i].x,
                   Y_Normal_Vector->points[i].y,
                   Y_Normal_Vector->points[i].z);

        if(k.dot(h) < 0)
        {
            Y_Normal_Vector->points[i].x = -Y_Normal_Vector->points[i].x;
            Y_Normal_Vector->points[i].y = -Y_Normal_Vector->points[i].y;
            Y_Normal_Vector->points[i].z = -Y_Normal_Vector->points[i].z;
        }

        Vector3d v(Y_Normal_Vector->points[i].x, Y_Normal_Vector->points[i].y, Y_Normal_Vector->points[i].z);
        Vector3d w(Z_Normal_Vector->points[i].x, Z_Normal_Vector->points[i].y, Z_Normal_Vector->points[i].z);
        Vector3d u = v.cross(w);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ x_normal_vector;
        x_normal_vector.x = u[0] / n;
        x_normal_vector.y = u[1] / n;
        x_normal_vector.z = u[2] / n; 
        
        X_Normal_Vector->points.push_back(x_normal_vector);
    }
    std::cout << "X_Normal_Vector: " << X_Normal_Vector->points.size() << std::endl;
    std::cout << "Y_Normal_Vector: " << Y_Normal_Vector->points.size() << std::endl;
    std::cout << "Z_Normal_Vector: " << Z_Normal_Vector->points.size() << std::endl  << std::endl;
}

void Temp_vector_computation( Cloud::Ptr Temp_vector1,
                              Cloud::Ptr Temp_vector2,
                              Cloud::Ptr PathPoint_Position_final, 
                              Cloud::Ptr Torch_Normal_Vector_final)
{
    for(std::size_t i = 0; i < PathPoint_Position_final->points.size()-1; i++)
    {
        pcl::PointXYZ temp_vector;

        temp_vector.x = PathPoint_Position_final->points[i+1].x - PathPoint_Position_final->points[i].x;
        temp_vector.y = PathPoint_Position_final->points[i+1].y - PathPoint_Position_final->points[i].y;
        temp_vector.z = PathPoint_Position_final->points[i+1].z - PathPoint_Position_final->points[i].z;

        Vector3d v(                  temp_vector.x,                   temp_vector.y,                   temp_vector.z);
        Vector3d w(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d u = w.cross(v);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ vec;
        vec.x = u[0] / n;
        vec.y = u[1] / n;
        vec.z = u[2] / n; 
        
        Temp_vector1->points.push_back(vec);

        if(i == PathPoint_Position_final->points.size()-1-1)
        {
            Temp_vector1->points.push_back(vec);
        }
    }
    std::cout << "Temp_vector1: " << Temp_vector1->points.size() << std::endl << std::endl;

    for(std::size_t i = 0; i < Temp_vector1->points.size(); i++)
    {
        Vector3d v(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d w(              Temp_vector1->points[i].x,               Temp_vector1->points[i].y,               Temp_vector1->points[i].z);
        Vector3d u = w.cross(v);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ vec;
        vec.x = u[0] / n;
        vec.y = u[1] / n;
        vec.z = u[2] / n; 
        Temp_vector2->points.push_back(vec);
    }
    std::cout << "Temp_vector2: " << Temp_vector2->points.size() << std::endl << std::endl;
}

void Y_Normal_Vector_computation(Cloud::Ptr Y_Normal_Vector,
                                 Cloud::Ptr Temp_vector1,
                                 Cloud::Ptr Temp_vector2)
{
    for(std::size_t i = 0; i < Temp_vector1->points.size(); i++)
    {
        Vector3d k(0, -1, 0);

        Vector3d vec1(Temp_vector1->points[i].x,
                     Temp_vector1->points[i].y,
                     Temp_vector1->points[i].z);

        Vector3d vec2(Temp_vector2->points[i].x,
                     Temp_vector2->points[i].y,
                     Temp_vector2->points[i].z);

        if( fabs(k.dot(vec1)) >= fabs(k.dot(vec2)) )
        {
            Y_Normal_Vector->points.push_back(Temp_vector1->points[i]);
        }
        else
        {
            Y_Normal_Vector->points.push_back(Temp_vector2->points[i]);
        }
    }
    std::cout << "Y_Normal_Vector: " << Y_Normal_Vector->points.size() << std::endl << std::endl;
}

void X_Z_Normal_Vector_computation(Cloud::Ptr Y_Normal_Vector, 
                                   Cloud::Ptr Z_Normal_Vector, 
                                   Cloud::Ptr X_Normal_Vector, 
                                   Cloud::Ptr Torch_Normal_Vector_final)
{
    for(std::size_t i = 0; i < Y_Normal_Vector->points.size(); i++)
    {
        Vector3d v(-Torch_Normal_Vector_final->points[i].x, -Torch_Normal_Vector_final->points[i].y, -Torch_Normal_Vector_final->points[i].z);
        Vector3d w(           Y_Normal_Vector->points[i].x,            Y_Normal_Vector->points[i].y,            Y_Normal_Vector->points[i].z);
        Vector3d u = w.cross(v);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ x_normal_vector;
        x_normal_vector.x = u[0] / n;
        x_normal_vector.y = u[1] / n;
        x_normal_vector.z = u[2] / n; 
        X_Normal_Vector->points.push_back(x_normal_vector);

        pcl::PointXYZ z_normal_vector;
        z_normal_vector.x = -Torch_Normal_Vector_final->points[i].x;
        z_normal_vector.y = -Torch_Normal_Vector_final->points[i].y;
        z_normal_vector.z = -Torch_Normal_Vector_final->points[i].z;
        
        Z_Normal_Vector->points.push_back(z_normal_vector);
    }
    std::cout << "X_Normal_Vector: " << X_Normal_Vector->points.size()  << std::endl;
    std::cout << "Z_Normal_Vector: " << Z_Normal_Vector->points.size()  << std::endl  << std::endl;
}

void direction_modification(Cloud::Ptr X_Normal_Vector,
                            Cloud::Ptr Y_Normal_Vector, 
                            Cloud::Ptr Z_Normal_Vector, 
                            Cloud::Ptr PathPoint_Position_final)
{
    X_Normal_Vector->points.clear();
    for(std::size_t i = 0; i < Y_Normal_Vector->points.size(); i++)
    {
        Vector3d k(0, -1, 0);

        Vector3d h(Y_Normal_Vector->points[i].x,
                   Y_Normal_Vector->points[i].y,
                   Y_Normal_Vector->points[i].z);

        if(k.dot(h) < 0)
        {
            Y_Normal_Vector->points[i].x = -Y_Normal_Vector->points[i].x;
            Y_Normal_Vector->points[i].y = -Y_Normal_Vector->points[i].y;
            Y_Normal_Vector->points[i].z = -Y_Normal_Vector->points[i].z;
        }

        Vector3d v(Y_Normal_Vector->points[i].x, Y_Normal_Vector->points[i].y, Y_Normal_Vector->points[i].z);
        Vector3d w(Z_Normal_Vector->points[i].x, Z_Normal_Vector->points[i].y, Z_Normal_Vector->points[i].z);
        Vector3d u = v.cross(w);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ x_normal_vector;
        x_normal_vector.x = u[0] / n;
        x_normal_vector.y = u[1] / n;
        x_normal_vector.z = u[2] / n; 
        
        X_Normal_Vector->points.push_back(x_normal_vector);
    }
    std::cout << "X_Normal_Vector: " << X_Normal_Vector->points.size() << std::endl;
    std::cout << "Y_Normal_Vector: " << Y_Normal_Vector->points.size() << std::endl;
    std::cout << "Z_Normal_Vector: " << Z_Normal_Vector->points.size() << std::endl  << std::endl;
}

std::vector< geometry_msgs::msg::Pose > 
Moveit_Pose_generation(float trajectory_point_size, 
                      Cloud::Ptr X_Normal_Vector, 
                      Cloud::Ptr Y_Normal_Vector, 
                      Cloud::Ptr Z_Normal_Vector, 
                      Cloud::Ptr PathPoint_Position_final)
{
    std::vector< geometry_msgs::msg::Pose > Rviz_TrajectoryPose;
    for(int i = 0; i < (int)trajectory_point_size; i++)
    {
        geometry_msgs::msg::Pose pose;

        Eigen::Matrix3d origin_base;
        origin_base << 1, 0, 0,
                       0, 1, 0, 
                       0, 0, 1;
        
        Eigen::Matrix3d transformed_base;
        transformed_base << X_Normal_Vector->points[i].x, Y_Normal_Vector->points[i].x, Z_Normal_Vector->points[i].x,
                            X_Normal_Vector->points[i].y, Y_Normal_Vector->points[i].y, Z_Normal_Vector->points[i].y, 
                            X_Normal_Vector->points[i].z, Y_Normal_Vector->points[i].z, Z_Normal_Vector->points[i].z;

        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = origin_base * transformed_base;
        
        Eigen::Quaterniond rotation(rotation_matrix);

        pose.position.x = PathPoint_Position_final->points[i].x;
        pose.position.y = PathPoint_Position_final->points[i].y;
        pose.position.z = PathPoint_Position_final->points[i].z;
        pose.orientation.x = rotation.x();
        pose.orientation.y = rotation.y();
        pose.orientation.z = rotation.z();
        pose.orientation.w = rotation.w();
        Rviz_TrajectoryPose.push_back(pose);
    }
    return Rviz_TrajectoryPose;
}

void URx_Pose_generation( float trajectory_point_size, 
                          Cloud::Ptr X_Normal_Vector, 
                          Cloud::Ptr Y_Normal_Vector, 
                          Cloud::Ptr Z_Normal_Vector, 
                          Cloud::Ptr PathPoint_Position_final,
                          std::vector< geometry_msgs::msg::Pose > &Welding_Trajectory)
{
    for(int i = 0; i < (int)trajectory_point_size; i++)
    {
        Eigen::Matrix3d origin_base_URx;
        origin_base_URx << -1,  0, 0,
                            0, -1, 0, 
                            0,  0, 1;
        
        Eigen::Matrix3d transformed_base_URx;
        transformed_base_URx << X_Normal_Vector->points[i].x, Y_Normal_Vector->points[i].x, Z_Normal_Vector->points[i].x,
                                X_Normal_Vector->points[i].y, Y_Normal_Vector->points[i].y, Z_Normal_Vector->points[i].y, 
                                X_Normal_Vector->points[i].z, Y_Normal_Vector->points[i].z, Z_Normal_Vector->points[i].z;

        Eigen::Matrix3d rotation_matrix_URx;
        rotation_matrix_URx = origin_base_URx * transformed_base_URx;

        Matrix4d T_Base2Torch;
        T_Base2Torch << rotation_matrix_URx(0,0), rotation_matrix_URx(0,1), rotation_matrix_URx(0,2), -PathPoint_Position_final->points[i].x,
                        rotation_matrix_URx(1,0), rotation_matrix_URx(1,1), rotation_matrix_URx(1,2), -PathPoint_Position_final->points[i].y,
                        rotation_matrix_URx(2,0), rotation_matrix_URx(2,1), rotation_matrix_URx(2,2),  PathPoint_Position_final->points[i].z,
                        0                      , 0                       , 0                       , 1;

        Matrix4d T_End2Torch;
        T_End2Torch << 1,  0,        0,        0,
                       0,  0.660006, 0.751261, 0.048,
                      -0, -0.751261, 0.660006, 0.227,
                       0,  0,        0,        1;

        Matrix4d T_Base2End = T_Base2Torch * T_End2Torch.inverse();

        Eigen::Matrix3d End_rotation_matrix_URx;
        End_rotation_matrix_URx << T_Base2End(0,0), T_Base2End(0,1), T_Base2End(0,2),
                                   T_Base2End(1,0), T_Base2End(1,1), T_Base2End(1,2),
                                   T_Base2End(2,0), T_Base2End(2,1), T_Base2End(2,2);

        Eigen::AngleAxisd End_rotation_vector_URx(End_rotation_matrix_URx);

        geometry_msgs::msg::Pose pose;
        pose.position.x    = T_Base2End(0,3);
        pose.position.y    = T_Base2End(1,3);
        pose.position.z    = T_Base2End(2,3);
        pose.orientation.x = End_rotation_vector_URx.angle() * End_rotation_vector_URx.axis().x(); 
        pose.orientation.y = End_rotation_vector_URx.angle() * End_rotation_vector_URx.axis().y(); 
        pose.orientation.z = End_rotation_vector_URx.angle() * End_rotation_vector_URx.axis().z(); 
        pose.orientation.w = 0;
        Welding_Trajectory.push_back(pose);
    }
}

Eigen::Quaterniond rotation_Quaternionslerp(Eigen::Quaterniond starting, Eigen::Quaterniond ending, float t )
{
    float cosa = starting.x()*ending.x() + starting.y()*ending.y() + starting.z()*ending.z() + starting.w()*ending.w();
    
    Eigen::Quaterniond ending_copy = ending;
    if ( cosa < 0.0f ) 
    {
        ending_copy.x() = -ending.x();
        ending_copy.y() = -ending.y();
        ending_copy.z() = -ending.z();
        ending_copy.w() = -ending.w();
        cosa = -cosa;
    }
    
    float k0 = 0, k1 = 0;
    if ( cosa > 0.9995f ) 
    {
        k0 = 1.0f - t;
        k1 = t;
    }
    else 
    {
        float sina = sqrt( 1.0f - cosa*cosa );
        float a = atan2( sina, cosa );
        k0 = sin((1.0f - t)*a)  / sina;
        k1 = sin(t*a) / sina;
    }

    Eigen::Quaterniond result;
    result.x() = starting.x()*k0 + ending_copy.x()*k1;
    result.y() = starting.y()*k0 + ending_copy.y()*k1;
    result.z() = starting.z()*k0 + ending_copy.z()*k1;
    result.w() = starting.w()*k0 + ending_copy.w()*k1;

    return result;
}

void orientation_definition_verticalType( Cloud::Ptr X_Normal_Vector,
                                          Cloud::Ptr Y_Normal_Vector, 
                                          Cloud::Ptr Z_Normal_Vector, 
                                          Cloud::Ptr PathPoint_Position_final, 
                                          Cloud::Ptr Torch_Normal_Vector_final)
{
    X_Normal_Vector_vertical( X_Normal_Vector,
                              PathPoint_Position_final, 
                              Torch_Normal_Vector_final);

    Y_Z_Normal_Vector_vertical( Y_Normal_Vector, 
                                Z_Normal_Vector, 
                                X_Normal_Vector, 
                                Torch_Normal_Vector_final);

    direction_modification_vertical(X_Normal_Vector, 
                                    Y_Normal_Vector, 
                                    Z_Normal_Vector, 
                                    PathPoint_Position_final);
}

void orientation_definition_horizontalType( Cloud::Ptr X_Normal_Vector,
                                            Cloud::Ptr Y_Normal_Vector, 
                                            Cloud::Ptr Z_Normal_Vector, 
                                            Cloud::Ptr PathPoint_Position_final, 
                                            Cloud::Ptr Torch_Normal_Vector_final)
{
    Y_Normal_Vector_horizontal( Y_Normal_Vector,
                                PathPoint_Position_final, 
                                Torch_Normal_Vector_final);

    X_Z_Normal_Vector_horizontal( Y_Normal_Vector, 
                                  Z_Normal_Vector, 
                                  X_Normal_Vector, 
                                  Torch_Normal_Vector_final);

    direction_modification_horizontal(X_Normal_Vector, 
                                      Y_Normal_Vector, 
                                      Z_Normal_Vector, 
                                      PathPoint_Position_final);
}

void orientation_computation( Cloud::Ptr X_Normal_Vector,
                              Cloud::Ptr Y_Normal_Vector, 
                              Cloud::Ptr Z_Normal_Vector, 
                              Cloud::Ptr PathPoint_Position_final, 
                              Cloud::Ptr Torch_Normal_Vector_final)
{
    Cloud::Ptr Temp_vector1 (new Cloud);   
    Cloud::Ptr Temp_vector2 (new Cloud); 

    Temp_vector_computation(Temp_vector1,
                            Temp_vector2,
                            PathPoint_Position_final,
                            Torch_Normal_Vector_final);

    Y_Normal_Vector_computation(Y_Normal_Vector,
                                Temp_vector1, 
                                Temp_vector2);

    X_Z_Normal_Vector_computation(Y_Normal_Vector, 
                                  Z_Normal_Vector, 
                                  X_Normal_Vector, 
                                  Torch_Normal_Vector_final);

    int intern = 3;
    Cloud::Ptr X_Normal_Vector_filtered (new Cloud); 
  
    for(std::size_t i = 0; i < X_Normal_Vector->points.size(); i++)
    {
        pcl::PointXYZ x_normal_vector_filtered;

        if(i >= (std::size_t)intern && i <= X_Normal_Vector->points.size() - 1 - intern)
        {
            for(int j = i - intern; j <= i + intern; j++)
            {
                x_normal_vector_filtered.x += X_Normal_Vector->points[j].x;
                x_normal_vector_filtered.y += X_Normal_Vector->points[j].y;
                x_normal_vector_filtered.z += X_Normal_Vector->points[j].z;
            }
            x_normal_vector_filtered.x /= (2 * intern + 1.0);
            x_normal_vector_filtered.y /= (2 * intern + 1.0);
            x_normal_vector_filtered.z /= (2 * intern + 1.0);
        }
        else if(i < (std::size_t)intern)
        {
            for(int j = 0; j <= 0 + 2 * intern; j++)
            {
                x_normal_vector_filtered.x += X_Normal_Vector->points[j].x;
                x_normal_vector_filtered.y += X_Normal_Vector->points[j].y;
                x_normal_vector_filtered.z += X_Normal_Vector->points[j].z;
            }
            x_normal_vector_filtered.x /= (2 * intern + 1.0);
            x_normal_vector_filtered.y /= (2 * intern + 1.0);
            x_normal_vector_filtered.z /= (2 * intern + 1.0);
        }
        else
        {
            for(int j = X_Normal_Vector->points.size() - 1 - 2 * intern; j <= (int)X_Normal_Vector->points.size() - 1; j++)
            {
                x_normal_vector_filtered.x += X_Normal_Vector->points[j].x;
                x_normal_vector_filtered.y += X_Normal_Vector->points[j].y;
                x_normal_vector_filtered.z += X_Normal_Vector->points[j].z;
            }
            x_normal_vector_filtered.x /= (2 * intern + 1.0);
            x_normal_vector_filtered.y /= (2 * intern + 1.0);
            x_normal_vector_filtered.z /= (2 * intern + 1.0);
        }
        X_Normal_Vector_filtered->points.push_back(x_normal_vector_filtered);
    }

    X_Normal_Vector->points.clear();
    for(std::size_t i = 0; i < X_Normal_Vector_filtered->points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = X_Normal_Vector_filtered->points[i].x;
        p.y = X_Normal_Vector_filtered->points[i].y;
        p.z = X_Normal_Vector_filtered->points[i].z;
        X_Normal_Vector->points.push_back(p);
    }

    Y_Normal_Vector->points.clear();
    for(std::size_t i = 0; i < X_Normal_Vector->points.size(); i++)
    {
        Vector3d v(X_Normal_Vector->points[i].x, X_Normal_Vector->points[i].y, X_Normal_Vector->points[i].z);
        Vector3d w(Z_Normal_Vector->points[i].x, Z_Normal_Vector->points[i].y, Z_Normal_Vector->points[i].z);
        Vector3d u = w.cross(v);
        float n = sqrt( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] );

        pcl::PointXYZ y_normal_vector;
        y_normal_vector.x = u[0] / n;
        y_normal_vector.y = u[1] / n;
        y_normal_vector.z = u[2] / n; 
        
        Y_Normal_Vector->points.push_back(y_normal_vector);
    }

    direction_modification( X_Normal_Vector, 
                            Y_Normal_Vector, 
                            Z_Normal_Vector, 
                            PathPoint_Position_final);
}



std::vector<geometry_msgs::msg::Pose>
Ultimate_6DOF_TrajectoryGeneration(std::vector< geometry_msgs::msg::Pose > &Welding_Trajectory,
                                   Cloud::Ptr PathPoint_Position,
                                   std::vector<cv::Point3f> Torch_Normal_Vector)
{
    // 1. 把 cv::Point3f 转成 Eigen::Vector3f（你的内部函数需要这个！）
    std::vector<Eigen::Vector3f> torch_vec;
    for (auto& cv_p : Torch_Normal_Vector) {
        Eigen::Vector3f eigen_p;
        eigen_p.x() = cv_p.x;
        eigen_p.y() = cv_p.y;
        eigen_p.z() = cv_p.z;
        torch_vec.push_back(eigen_p);
    }

    int points_cut_count = 3;
    Cloud::Ptr PathPoint_Position_final(new Cloud);
    Cloud::Ptr Torch_Normal_Vector_final(new Cloud);

    // 2. 传入转换后的 Eigen::Vector3f
    pathpoint_cut_head_tail(points_cut_count,
                            PathPoint_Position,
                            torch_vec,  
                            PathPoint_Position_final,
                            Torch_Normal_Vector_final);

    Cloud::Ptr X_Normal_Vector(new Cloud);
    Cloud::Ptr Y_Normal_Vector(new Cloud);
    Cloud::Ptr Z_Normal_Vector(new Cloud);

    orientation_computation(X_Normal_Vector,
                            Y_Normal_Vector,
                            Z_Normal_Vector,
                            PathPoint_Position_final,
                            Torch_Normal_Vector_final);

    float trajectory_point_size = PathPoint_Position_final->size();

    std::vector<geometry_msgs::msg::Pose> Rviz_TrajectoryPose =
        Moveit_Pose_generation(trajectory_point_size,
                               X_Normal_Vector,
                               Y_Normal_Vector,
                               Z_Normal_Vector,
                               PathPoint_Position_final);

    URx_Pose_generation(trajectory_point_size,
                        X_Normal_Vector,
                        Y_Normal_Vector,
                        Z_Normal_Vector,
                        PathPoint_Position_final,
                        Welding_Trajectory);

    return Rviz_TrajectoryPose;
}
