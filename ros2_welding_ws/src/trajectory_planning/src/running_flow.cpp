// ============================================
// running_flow.cpp - ROS2版本
// 修改点：所有ROS1头文件替换为ROS2版本
// ============================================

#include <running_flow.hpp>
#include "transformation.hpp"
#include "pc2r_welding_generator.hpp"
#include "welding_config.hpp"

#include <rclcpp/rclcpp.hpp>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>
#include <dirent.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

using namespace cv;
using namespace std;
using namespace Eigen;

// 全局变量（ROS2版本）
vector<geometry_msgs::msg::Pose> Rviz_TrajectoryPose;
vector<pcl::PointXYZ> all_realsense_position;

// ============================================
// 修改点1：CAD_TrajectoryPlanning函数
// 发布者类型改为SharedPtr，消息类型改为msg::
// ============================================
vector<geometry_msgs::msg::Pose> CAD_TrajectoryPlanning(
    Cloud::Ptr cloud_ptr,
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher, 
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime)
{
    input_pointcloud_filter(cloud_ptr);

    pcl::PointXYZ singleFrame_realsense_position;
    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = cloud_ptr->points[i].x;
        p.y = cloud_ptr->points[i].y;
        p.z = cloud_ptr->points[i].z;

        singleFrame_realsense_position.x += p.x;    
        singleFrame_realsense_position.y += p.y;    
        singleFrame_realsense_position.z += p.z;    
    }
    singleFrame_realsense_position.x = singleFrame_realsense_position.x / cloud_ptr->points.size();
    singleFrame_realsense_position.y = singleFrame_realsense_position.y / cloud_ptr->points.size();
    singleFrame_realsense_position.z = singleFrame_realsense_position.z / cloud_ptr->points.size();

    welding_seam_location(cloud_ptr, 
                          singleFrame_realsense_position, 
                          pub_pointcloud, 
                          pointcloud_publisher);

    singleFrame_realsense_position.x = -0.260405;
    singleFrame_realsense_position.y =  0.700942;
    singleFrame_realsense_position.z =  0.238146;
    all_realsense_position.push_back(singleFrame_realsense_position);

    singleFrame_realsense_position.x = -0.0355285;
    singleFrame_realsense_position.y =  0.701815;
    singleFrame_realsense_position.z =  0.345082;
    all_realsense_position.push_back(singleFrame_realsense_position);

    singleFrame_realsense_position.x =  0.176993;
    singleFrame_realsense_position.y =  0.702516;
    singleFrame_realsense_position.z =  0.269644;
    all_realsense_position.push_back(singleFrame_realsense_position);

    cout << "all_realsense_position.size(): " << all_realsense_position.size() << endl;
    Rviz_TrajectoryPose = trajectory_planning(cloud_ptr, 
                                              all_realsense_position,
                                              pub_pointcloud, 
                                              pointcloud_publisher,   
                                              Welding_Trajectory_publisher,
                                              naptime);

    return Rviz_TrajectoryPose;
}

// ============================================
// 修改点2：trajectory_6DOF_generation函数
// ============================================
vector<geometry_msgs::msg::Pose> trajectory_6DOF_generation(
    pcl::PointXYZ singleFrame_realsense_position, 
    Cloud::Ptr cloud_ptr, 
    Cloud::Ptr cloud_ptr_modelSeam, 
    bool& trajectoryPlanning_flag, 
    int& receive_capture_count,
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher, 
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime)
{
    if (trajectoryPlanning_flag == true)
    {
        trajectoryPlanning_flag = false;
        receive_capture_count++;

        input_pointcloud_filter(cloud_ptr);

        welding_seam_location(cloud_ptr, 
                              singleFrame_realsense_position, 
                              pub_pointcloud, 
                              pointcloud_publisher);

        integrate_allsingle_pointcloudFrame(cloud_ptr, 
                                            cloud_ptr_modelSeam, 
                                            pub_pointcloud, 
                                            pointcloud_publisher);

        cloud_ptr_modelSeam->width = 1;
        cloud_ptr_modelSeam->height = cloud_ptr_modelSeam->points.size();

        cout << "cloud_ptr_modelSeam->points.size()" << cloud_ptr_modelSeam->points.size() << endl;
        pcl::PCDWriter writer;
        if (cloud_ptr_modelSeam->points.size() > 0)
            writer.write("/home/someityhuihui/ros2_welding_test_data/seam.pcd", *cloud_ptr_modelSeam, false);

        all_realsense_position.push_back(singleFrame_realsense_position);
        cout << "all_realsense_position: " << all_realsense_position.size() << endl;
        Rviz_TrajectoryPose = trajectory_planning(cloud_ptr_modelSeam, 
                                                  all_realsense_position,
                                                  pub_pointcloud, 
                                                  pointcloud_publisher,   
                                                  Welding_Trajectory_publisher,
                                                  naptime);
    }

    return Rviz_TrajectoryPose;
}

// ============================================
// 修改点3：welding_seam_location函数
// ============================================
bool welding_seam_location(Cloud::Ptr cloud_ptr, 
                           pcl::PointXYZ realsense_position, 
                           sensor_msgs::msg::PointCloud2 pub_pointcloud, 
                           rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    float seam_detection_radius = 0.01;
    clock_t begin = clock();

    int show_Pointcloud_timeMax = 100;

    PointCloud::Ptr cloud_ptr_show(new PointCloud);

    cout << "要不要处理这帧点云" << endl;
    bool process_flag = processing_frame_ornot(cloud_ptr, 
                                                show_Pointcloud_timeMax, 
                                                cloud_ptr_show, 
                                                pub_pointcloud, 
                                                pointcloud_publisher);
    if (!process_flag)
    {
        cloud_ptr->points.clear();
        return false;
    }

    // Seam Location:
    cout << endl << "1.读入原始pointcloud" << endl << endl;
    SurfaceProfile_Reconstruction(seam_detection_radius, 
                                  cloud_ptr, 
                                  cloud_ptr_show);
    Cloud::Ptr cloud_ptr_origin = cloud_ptr_origin_copy(cloud_ptr);

    show_pointcloud_Rviz(show_Pointcloud_timeMax,   
                          cloud_ptr_show, 
                          pub_pointcloud, 
                          pointcloud_publisher);

    cout << "2.算出所有点的法向量" << endl << endl;
    Point3f Cam_Position; 
    Cam_Position.x = realsense_position.x; 
    Cam_Position.y = realsense_position.y; 
    Cam_Position.z = realsense_position.z; 
    vector<Point3f> Normal = PointNormal_Computation(seam_detection_radius, 
                                                     cloud_ptr, 
                                                     cloud_ptr_show, 
                                                     Cam_Position);
    show_pointcloud_Rviz(show_Pointcloud_timeMax, 
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << "3.分割出所有可能的焊接缝" << endl << endl;
    Delete_SmoothChange_Plane(seam_detection_radius, 
                              cloud_ptr, 
                              cloud_ptr_show, 
                              Normal, 
                              pub_pointcloud, 
                              pointcloud_publisher);

    show_pointcloud_Rviz(show_Pointcloud_timeMax, 
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << "4.人工筛选出目标焊接缝" << endl << endl;
    Screen_Candidate_Seam(cloud_ptr, 
                          cloud_ptr_show, 
                          pub_pointcloud, 
                          pointcloud_publisher);
    show_pointcloud_Rviz(show_Pointcloud_timeMax, 
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << endl;
    clock_t end = clock();
    double elapsed_secs = static_cast<double>(end - begin) / CLOCKS_PER_SEC;
    cout << elapsed_secs << " s" << endl;
    
    return true;
}

// ============================================
// 修改点4：trajectory_planning函数
// ============================================
// ============================================
// 修改后的 trajectory_planning 函数
// 支持选择原始算法或论文算法
// ============================================

vector<geometry_msgs::msg::Pose> trajectory_planning(
    Cloud::Ptr cloud_ptr_modelSeam,
    vector<pcl::PointXYZ> all_realsense_position,
    sensor_msgs::msg::PointCloud2 pub_pointcloud, 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher,
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr Welding_Trajectory_publisher,
    rclcpp::Rate& naptime)
{
    cout << endl << endl << endl;
    cout << "========================================" << endl;
    cout << "Select algorithm:" << endl;
    cout << "  1 - Original algorithm (your implementation)" << endl;
    cout << "  2 - PC2R algorithm (IBNN-LM-ICP + EI Response)" << endl;
    cout << "========================================" << endl;
    cout << "Enter choice (1 or 2): ";
    
    string algo_choice;
    cin >> algo_choice;
    
    // ============================================
    // 选项2: 使用论文算法 (PC2R)
    // ============================================
    if (algo_choice == "2")
    {
        cout << endl << "Using PC2R (Point Cloud to Robotic Welding) Algorithm..." << endl;
        cout << "Based on IBNN-LM-ICP point cloud stitching and EI response edge detection" << endl;
        cout << "Reference: Liu et al., Robotics and Computer-Integrated Manufacturing, 2025" << endl << endl;
        
        // 收集多视角点云数据
        // 注意：这里需要根据您的实际数据组织方式修改
        std::vector<CloudPtr> multi_view_clouds;
        std::vector<Eigen::Vector3f> camera_poses;
        
        // 方法1：如果 cloud_ptr_modelSeam 已经是融合后的点云，直接使用
        // 方法2：如果需要重新拼接，从原始数据读取
        
        cout << "Do you want to re-stitch from multi-view data? (yes/no): ";
        string re_stitch;
        cin >> re_stitch;
        
        if (re_stitch == "yes" || re_stitch == "y")
        {
            // 从数据集文件夹读取所有点云帧
            string base_path = "/home/someityhuihui/ros2_welding_test_data/seam_3_pipe_normalized_irregular_dataset_5_frame/";
            int num_frames = count_pointcloud_frameNum(base_path);
            
            cout << "Found " << num_frames << " point cloud frames" << endl;
            
            for (int frame = 1; frame <= num_frames; frame++)
            {
                ostringstream stream;
                stream << frame;
                string ith_frame = stream.str();
                string file_path = base_path + "/" + ith_frame + "_frame.pcd";
                
                CloudPtr cloud_frame(new Cloud);
                pcl::PCDReader reader;
                if (reader.read(file_path, *cloud_frame) >= 0)
                {
                    multi_view_clouds.push_back(cloud_frame);
                    
                    // 读取相机位姿
                    string pose_file = base_path + "/" + ith_frame + "_frame.txt";
                    ifstream fin(pose_file.c_str());
                    if (fin)
                    {
                        string s;
                        getline(fin, s);
                        int _1_T = s.find("\t", 0);
                        int _2_T = s.find("\t", _1_T + 1);
                        
                        Eigen::Vector3f cam_pose;
                        stringstream ss_cam_x, ss_cam_y, ss_cam_z;
                        ss_cam_x << s.substr(0, _1_T);
                        ss_cam_y << s.substr(_1_T + 1, _2_T - _1_T - 1);
                        ss_cam_z << s.substr(_2_T + 1, s.length() - _2_T - 1);
                        ss_cam_x >> cam_pose.x();
                        ss_cam_y >> cam_pose.y();
                        ss_cam_z >> cam_pose.z();
                        camera_poses.push_back(cam_pose);
                    }
                    
                    cout << "Loaded frame " << frame << ": " << cloud_frame->points.size() << " points" << endl;
                }
            }
        }
        else
        {
            // 直接使用传入的点云
            multi_view_clouds.push_back(cloud_ptr_modelSeam);
            if (!all_realsense_position.empty())
            {
                camera_poses.push_back(Eigen::Vector3f(
                    all_realsense_position[0].x,
                    all_realsense_position[0].y,
                    all_realsense_position[0].z
                ));
            }
            else
            {
                camera_poses.push_back(Eigen::Vector3f(0, 0, 0.5f));
            }
        }
        
        if (multi_view_clouds.empty())
        {
            cout << "ERROR: No point cloud data available!" << endl;
            vector<geometry_msgs::msg::Pose> no_result;
            return no_result;
        }
        
        // 创建PC2R生成器
        PC2RWeldingGenerator generator;
        // generator.initialize(
        //     0.01f,   // 点云拼接精度
        //     0.01f,  // 边缘检测半径
        //     1.0f,    // 边缘阈值 epsilon
        //     3        // B样条次数
        // );
        
        // 替换 generator.initialize 调用为：
        generator.initialize(
            g_config.stitching.ibnn_epsilon,
            g_config.edge.search_radius,
            g_config.edge.epsilon,
            g_config.bspline.degree
        );
        
        // 生成焊接路径
        std::vector<WeldingPathPoint> welding_path = generator.generateWeldingPath(
            multi_view_clouds, 
            camera_poses,
            rclcpp::get_logger("trajectory_planning"),
            nullptr
        );
        
        if (welding_path.empty())
        {
            cout << "ERROR: Failed to generate welding path!" << endl;
            vector<geometry_msgs::msg::Pose> no_result;
            return no_result;
        }
        
        // 保存中间结果（可选）
        generator.saveIntermediateResults("/home/someityhuihui/ros2_welding_test_data/pc2r_results/");
        
        // 转换为ROS消息
        vector<geometry_msgs::msg::Pose> result;
        for (const auto& point : welding_path)
        {
            result.push_back(weldingPathPointToPose(point));
        }
        
        // 发布轨迹
        cout << "Publishing " << result.size() << " trajectory points..." << endl;
        for (size_t i = 0; i < result.size(); i++)
        {
            Welding_Trajectory_publisher->publish(result[i]);
            naptime.sleep();
            
            // 每10个点打印一次进度
            if (i % 10 == 0 || i == result.size() - 1)
            {
                cout << "  Published " << i+1 << "/" << result.size() << " points" << endl;
            }
        }
        
        // 保存到CSV文件
        ofstream outFile;
        outFile.open("/home/someityhuihui/ros2_welding_test_data/pc2r_trajectory.csv", ios::out);
        for (size_t i = 0; i < result.size(); i++)
        {
            outFile << result[i].position.x << ',' 
                    << result[i].position.y << ',' 
                    << result[i].position.z << ','
                    << result[i].orientation.x << ',' 
                    << result[i].orientation.y << ','
                    << result[i].orientation.z << ','
                    << result[i].orientation.w << endl;
        }
        outFile.close();
        
        cout << "Trajectory saved to pc2r_trajectory.csv" << endl;
        cout << "PC2R algorithm completed successfully!" << endl;
        
        return result;
    }
    
    // ============================================
    // 选项1: 原始算法
    // ============================================
    cout << endl << "Using Original Algorithm..." << endl;
    cout << "Generate the welding seam or not!!!!!!!!!! yes or xxxx ";
    string flag;
    cin >> flag;
    
    vector<geometry_msgs::msg::Pose> no;
    if (flag != "yes")
    {
        return no;
    }

    float seam_detection_radius = 0.01;
    int show_Pointcloud_timeMax = 100;
    PointCloud::Ptr cloud_ptr_show(new PointCloud);

    // Trajectory Planning:
    cout << "4.SurfaceProfile_Reconstruction" << endl << endl; 
    SurfaceProfile_Reconstruction(seam_detection_radius, 
                                  cloud_ptr_modelSeam, 
                                  cloud_ptr_show);
    show_pointcloud_Rviz(show_Pointcloud_timeMax,   
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << "5.提取焊接缝边界" << endl << endl;
    Cloud::Ptr seam_edge = Extract_Seam_edge(cloud_ptr_modelSeam, cloud_ptr_show);
    show_pointcloud_Rviz(show_Pointcloud_timeMax, 
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << "6.焊接缝三维轨迹点" << endl << endl; 
    Cloud::Ptr PathPoint_Position = PathPoint_Position_Generation(seam_edge, 
                                                                  cloud_ptr_modelSeam, 
                                                                  cloud_ptr_show);
    show_pointcloud_Rviz(show_Pointcloud_timeMax, 
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << "7.焊接缝轨迹点的焊枪方向" << endl << endl;
    // 注意：这里需要将 all_realsense_position 转换为 vector<pcl::PointXYZ>
    vector<Point3f> Torch_Normal_Vector = PathPoint_Orientation_Generation(PathPoint_Position, 
                                                                            cloud_ptr_modelSeam, 
                                                                            cloud_ptr_show, 
                                                                            all_realsense_position);
    show_pointcloud_Rviz(show_Pointcloud_timeMax, 
                         cloud_ptr_show, 
                         pub_pointcloud, 
                         pointcloud_publisher);

    cout << "8.生成最终6DOF轨迹" << endl << endl;
    vector<geometry_msgs::msg::Pose> Welding_Trajectory;
    vector<geometry_msgs::msg::Pose> Rviz_TrajectoryPose = Ultimate_6DOF_TrajectoryGeneration(Welding_Trajectory, 
                                                                                               PathPoint_Position, 
                                                                                               Torch_Normal_Vector);

    if (Welding_Trajectory.size() > 0)
    {
        for (size_t i = 0; i < Welding_Trajectory.size(); i++)
        {
            Welding_Trajectory_publisher->publish(Welding_Trajectory[i]);
            naptime.sleep();
        }
        cout << "3D path is published !!!!!!!!!" << endl;
    }

    ofstream outFile;
    outFile.open("/home/someityhuihui/ros2_welding_test_data/cad_test.csv", ios::out);
    for (size_t i = 0; i < Rviz_TrajectoryPose.size(); i++)
    {
        outFile << Rviz_TrajectoryPose[i].position.x << ',' 
                << Rviz_TrajectoryPose[i].position.y << ',' 
                << Rviz_TrajectoryPose[i].position.z << ','
                << Rviz_TrajectoryPose[i].orientation.x << ',' 
                << Rviz_TrajectoryPose[i].orientation.y << ','
                << Rviz_TrajectoryPose[i].orientation.z << ','
                << Rviz_TrajectoryPose[i].orientation.w << endl;
    }
    outFile.close();

    return Rviz_TrajectoryPose;
}

// ============================================
// input_pointcloud_filter函数（无需修改，无ROS依赖）
// ============================================
void input_pointcloud_filter(Cloud::Ptr cloud_ptr)
{
    Cloud::Ptr Cloud_filtered(new Cloud);
    float radius = 0.001;

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(radius, radius, radius);
    voxel.setInputCloud(cloud_ptr);
    voxel.filter(*Cloud_filtered);

    cout << "input pointcloud filtering done!!!" << endl;

    cloud_ptr->points.clear();
    for (size_t i = 0; i < Cloud_filtered->points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = Cloud_filtered->points[i].x;
        p.y = Cloud_filtered->points[i].y;
        p.z = Cloud_filtered->points[i].z;
        cloud_ptr->points.push_back(p);
    }
}

// ============================================
// integrate_allsingle_pointcloudFrame函数
// ============================================
void integrate_allsingle_pointcloudFrame(Cloud::Ptr cloud_ptr, 
                                         Cloud::Ptr cloud_ptr_modelSeam, 
                                         sensor_msgs::msg::PointCloud2 pub_pointcloud, 
                                         rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    int show_Pointcloud_timeMax = 100;

    for (size_t i = 0; i < cloud_ptr->points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = cloud_ptr->points[i].x;
        p.y = cloud_ptr->points[i].y;
        p.z = cloud_ptr->points[i].z;
        cloud_ptr_modelSeam->points.push_back(p);
    }
    cloud_ptr->points.clear();

    PointCloud::Ptr cloud_ptr_show(new PointCloud);
    for (size_t i = 0; i < cloud_ptr_modelSeam->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = cloud_ptr_modelSeam->points[i].x;
        p.y = cloud_ptr_modelSeam->points[i].y;
        p.z = cloud_ptr_modelSeam->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        cloud_ptr_show->points.push_back(p);
    }
    show_pointcloud_Rviz(show_Pointcloud_timeMax, cloud_ptr_show, pub_pointcloud, pointcloud_publisher);
}

// ============================================
// count_pointcloud_frameNum函数（无需修改）
// ============================================
int count_pointcloud_frameNum(string dataset_folder_path)
{
    DIR *dp;
    struct dirent *dirp;

    if ((dp = opendir(dataset_folder_path.c_str())) == NULL)
        cout << "Can't open " << dataset_folder_path << endl;

    int count = 0;
    while ((dirp = readdir(dp)) != NULL)
        count++;

    closedir(dp);

    cout << "pointcloud_frameNum: " << (count - 2) / 2 << endl;
    return (count - 2) / 2;
}

// ============================================
// build_model_pointcloud函数（无需修改）
// ============================================
void build_model_pointcloud(string dataset_folder_path, 
                            int pointcloud_frameNum,
                            PointCloud::Ptr model_pointcloud)
{
    Cloud::Ptr cloud_ptr_all(new Cloud);

    for (int receive_capture_count = 1; receive_capture_count <= pointcloud_frameNum; receive_capture_count++)
    {
        ostringstream stream;
        stream << receive_capture_count;
        string ith_frame = stream.str();

        string file_pointcloud_frame;
        file_pointcloud_frame = dataset_folder_path + "/" + ith_frame + "_frame.pcd";

        Cloud::Ptr cloud_ptr(new Cloud);
        pcl::PCDReader reader;
        reader.read(file_pointcloud_frame, *cloud_ptr);

        for (size_t i = 0; i < cloud_ptr->points.size(); i++)
        {
            pcl::PointXYZ p;
            p.x = cloud_ptr->points[i].x;
            p.y = cloud_ptr->points[i].y;
            p.z = cloud_ptr->points[i].z;
            cloud_ptr_all->points.push_back(p);
        }
    }

    Cloud::Ptr Cloud_filtered(new Cloud);
    float radius = 0.005;

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(radius, radius, radius);
    voxel.setInputCloud(cloud_ptr_all);
    voxel.filter(*Cloud_filtered);

    cout << "input pointcloud filtering done!!!" << endl;

    for (size_t i = 0; i < Cloud_filtered->points.size(); i++)
    {
        pcl::PointXYZRGB p;
        p.x = Cloud_filtered->points[i].x;
        p.y = Cloud_filtered->points[i].y;
        p.z = Cloud_filtered->points[i].z;
        p.b = 200;
        p.g = 200;
        p.r = 200;
        model_pointcloud->points.push_back(p);
    }

    cout << "model_pointcloud->points.size(): " << model_pointcloud->points.size() << endl << endl;
}

// ============================================
// read_realtime_pointcloud_frame函数（无需修改）
// ============================================
pcl::PointXYZ read_realtime_pointcloud_frame(string dataset_folder_path,
                                              int pointcloud_frameNum,
                                              int receive_capture_count,
                                              int& process_frame_count,
                                              bool& trajectoryPlanning_flag,
                                              Cloud::Ptr cloud_ptr)
{
    pcl::PointXYZ no;
    if (receive_capture_count > pointcloud_frameNum)
    {
        if (receive_capture_count > process_frame_count)
        {
            cout << endl << "process_frame_count:" << process_frame_count << "   no more pointcloud data!" << endl;
            process_frame_count = process_frame_count + 100;
        }
        return no;
    }

    pcl::PointXYZ realsense_position;

    ostringstream stream;
    stream << receive_capture_count;
    string ith_frame = stream.str();

    string file_realsense_position;
    file_realsense_position = dataset_folder_path + "/" + ith_frame + "_frame.txt";

    ifstream fin(file_realsense_position.c_str());
    if (fin)
    {
        string s;
        getline(fin, s);

        int _1_T = s.find("\t", 0);
        int _2_T = s.find("\t", _1_T + 1);

        string cam_x = s.substr(0, _1_T);
        stringstream ss_cam_x;
        ss_cam_x << cam_x;
        ss_cam_x >> realsense_position.x;

        string cam_y = s.substr(_1_T + 1, _2_T - _1_T - 1);
        stringstream ss_cam_y;
        ss_cam_y << cam_y;
        ss_cam_y >> realsense_position.y;

        string cam_z = s.substr(_2_T + 1, s.length() - _2_T - 1);
        stringstream ss_cam_z;
        ss_cam_z << cam_z;
        ss_cam_z >> realsense_position.z;

        cout << "read " << process_frame_count << "th pointcloud frame" << endl;
        cout << "realsense_position" << realsense_position << endl << endl;

        process_frame_count++;
        trajectoryPlanning_flag = true;
    }

    string file_pointcloud_frame;
    file_pointcloud_frame = dataset_folder_path + "/" + ith_frame + "_frame.pcd";

    pcl::PCDReader reader;
    reader.read(file_pointcloud_frame, *cloud_ptr);

    return realsense_position;
}

// ============================================
// read_trajectory_frame函数（无需修改）
// ============================================
vector<geometry_msgs::msg::Pose> read_trajectory_frame(string trajectoryInfo_folder_path)
{
    ifstream inFile(trajectoryInfo_folder_path.c_str());
    string lineStr;
    vector<vector<string>> strArray;
    while (getline(inFile, lineStr))
    {
        cout << lineStr << endl;
        stringstream ss(lineStr);
        string str;
        vector<string> lineArray;
        while (getline(ss, str, ','))
            lineArray.push_back(str);
        strArray.push_back(lineArray);
    }
    cout << "strArray.size(): " << strArray.size() << endl;

    vector<geometry_msgs::msg::Pose> trajectory_pose;
    for (size_t i = 0; i < strArray.size(); i++)
    {
        if (strArray[i].size() == 0)
        {
            cout << "empty!!" << endl;
            continue;
        }
        geometry_msgs::msg::Pose pose;

        stringstream position_x;
        position_x << strArray[i][0];
        position_x >> pose.position.x;

        stringstream position_y;
        position_y << strArray[i][1];
        position_y >> pose.position.y;

        stringstream position_z;
        position_z << strArray[i][2];
        position_z >> pose.position.z;

        stringstream orientation_x;
        orientation_x << strArray[i][3];
        orientation_x >> pose.orientation.x;

        stringstream orientation_y;
        orientation_y << strArray[i][4];
        orientation_y >> pose.orientation.y;

        stringstream orientation_z;
        orientation_z << strArray[i][5];
        orientation_z >> pose.orientation.z;

        stringstream orientation_w;
        orientation_w << strArray[i][6];
        orientation_w >> pose.orientation.w;

        cout << "pose: x=" << pose.position.x
     << " y=" << pose.position.y
     << " z=" << pose.position.z
     << " qx=" << pose.orientation.x
     << " qy=" << pose.orientation.y
     << " qz=" << pose.orientation.z
     << " qw=" << pose.orientation.w << endl;

        trajectory_pose.push_back(pose);
    }
    cout << "trajectory_pose.size(): " << trajectory_pose.size() << endl;

    return trajectory_pose;
}

// ============================================
// processing_frame_ornot函数
// ============================================
bool processing_frame_ornot(Cloud::Ptr cloud_ptr, 
                            int show_Pointcloud_timeMax, 
                            PointCloud::Ptr cloud_ptr_show, 
                            sensor_msgs::msg::PointCloud2 pub_pointcloud, 
                            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    cloud_ptr_show->clear();
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
    show_pointcloud_Rviz(show_Pointcloud_timeMax, cloud_ptr_show, pub_pointcloud, pointcloud_publisher);

    cout << endl << "process the pointcloud or not? yes or xxxx ";
    string flag;
    cin >> flag;
    bool process_flag = false;
    if (flag == "yes")
    {
        process_flag = true;
    }
    cloud_ptr_show->clear();
    cout << endl;

    return process_flag;
}

// ============================================
// 修改点5：show_pointcloud_Rviz函数
// 修改：ros::Time::now() -> rclcpp::Clock().now()
// 修改：pointcloud_publisher.publish -> ->publish
// ============================================
void show_pointcloud_Rviz(int show_Pointcloud_timeMax, 
                          PointCloud::Ptr show_Rviz_cloud, 
                          sensor_msgs::msg::PointCloud2 pub_pointcloud, 
                          rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
{
    for (float i = 0; i < show_Pointcloud_timeMax; i++)
    {
        pcl::toROSMsg(*show_Rviz_cloud, pub_pointcloud);
        pub_pointcloud.header.frame_id = "base_link";
        pub_pointcloud.header.stamp = rclcpp::Clock().now();  // 修改：ros::Time::now()
        pointcloud_publisher->publish(pub_pointcloud);       // 修改：-> 操作符
    }
    show_Rviz_cloud->points.clear();
}

// // ============================================
// // 修改点6：publish_pointcloud_Rviz函数
// // ============================================
// void publish_pointcloud_Rviz(string coordinate, 
//                              PointCloud::Ptr pointloud, 
//                              sensor_msgs::msg::PointCloud2 pub_pointcloud, 
//                              rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher)
// {
//     pcl::toROSMsg(*pointloud, pub_pointcloud);
//     pub_pointcloud.header.frame_id = coordinate;
//     pub_pointcloud.header.stamp = rclcpp::Clock().now();  // 修改：ros::Time::now()
//     pointcloud_publisher->publish(pub_pointcloud);       // 修改：-> 操作符
// }
