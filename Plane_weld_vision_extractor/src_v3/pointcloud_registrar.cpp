// pointcloud_registrar.cpp
#include "pointcloud_registrar.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>


PointCloudRegistrar::PointCloudRegistrar() : global_cloud_(new Cloud) {}

PointCloudRegistrar::~PointCloudRegistrar() {}



// 以下配合v1代码使用
bool PointCloudRegistrar::loadFrames(const std::string& folder_path, bool has_pose_files) {
    frames_.clear();
    camera_poses_.clear();
    
    std::cout << "[Registrar] 加载点云数据: " << folder_path << std::endl;
    
    int frame_id = 1;
    while (true) {
        std::string pcd_path = folder_path + "/" + std::to_string(frame_id) + "_frame.pcd";
        
        std::ifstream pcd_file(pcd_path);
        if (!pcd_file.good()) {
            break;
        }
        pcd_file.close();
        
        // 读取点云
        CloudPtr cloud(new Cloud);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) < 0) {
            std::cerr << "[Registrar] 无法读取: " << pcd_path << std::endl;
            break;
        }
        
        FrameData frame;
        frame.cloud = cloud;
        frame.id = frame_id;
        
        // 读取位姿（原始全局坐标）
        if (has_pose_files) {
            std::string pose_path = folder_path + "/" + std::to_string(frame_id) + "_frame.txt";
            std::ifstream pose_file(pose_path);
            if (pose_file.good()) {
                std::string line;
                std::getline(pose_file, line);
                std::replace(line.begin(), line.end(), ',', ' ');
                std::stringstream ss(line);
                ss >> frame.pose.x() >> frame.pose.y() >> frame.pose.z();
            } else {
                frame.pose = Eigen::Vector3f(0, 0, 0);
            }
        } else {
            frame.pose = Eigen::Vector3f(0, 0, 0);
        }
        
        frames_.push_back(frame);
        std::cout << "[Registrar] 加载帧 " << frame_id << ": " << cloud->points.size() 
                  << " 点, 位姿=(" << frame.pose.x() << "," << frame.pose.y() << "," << frame.pose.z() << ")" << std::endl;
        
        frame_id++;
    }
    
    std::cout << "[Registrar] 共加载 " << frames_.size() << " 帧" << std::endl;
    return !frames_.empty();
}

// 以下配合v2代码使用
// bool PointCloudRegistrar::loadFrames(const std::string& folder_path, bool has_pose_files) {
//     frames_.clear();
//     camera_poses_.clear();
    
//     std::cout << "[Registrar] 加载点云数据: " << folder_path << std::endl;
    
//     // 查找所有点云文件
//     int frame_id = 1;
//     while (true) {
//         std::string pcd_path = folder_path + "/" + std::to_string(frame_id) + "_frame.pcd";
        
//         std::ifstream pcd_file(pcd_path);
//         if (!pcd_file.good()) {
//             break;
//         }
//         pcd_file.close();
        
//         // 读取点云
//         CloudPtr cloud(new Cloud);
//         if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) < 0) {
//             std::cerr << "[Registrar] 无法读取: " << pcd_path << std::endl;
//             break;
//         }
        
//         std::cout << "[Registrar] 加载帧 " << frame_id << ": " << cloud->points.size() << " 点" << std::endl;
        
//         FrameData frame;
//         frame.cloud = cloud;
//         frame.id = frame_id;
        
//         // 读取位姿（如果有）
//         if (has_pose_files) {
//             std::string pose_path = folder_path + "/" + std::to_string(frame_id) + "_frame.txt";
//             std::ifstream pose_file(pose_path);
//             if (pose_file.good()) {
//                 std::string line;
//                 std::getline(pose_file, line);
//                 // 替换逗号为空格
//                 std::replace(line.begin(), line.end(), ',', ' ');
//                 std::stringstream ss(line);
//                 ss >> frame.pose.x() >> frame.pose.y() >> frame.pose.z();
//                 std::cout << "[Registrar]   原始位姿: (" << frame.pose.x() << ", " 
//                           << frame.pose.y() << ", " << frame.pose.z() << ")" << std::endl;
//             } else {
//                 std::cerr << "[Registrar]   警告: 找不到位姿文件 " << pose_path << std::endl;
//                 frame.pose = Eigen::Vector3f(0, 0, 0);
//             }
//         } else {
//             frame.pose = Eigen::Vector3f(0, 0, 0);
//         }
        
//         frames_.push_back(frame);
//         frame_id++;
//     }
    
//     std::cout << "[Registrar] 共加载 " << frames_.size() << " 帧" << std::endl;
//     return !frames_.empty();
// }

CloudPtr PointCloudRegistrar::downsample(CloudPtr cloud) {
    CloudPtr downsampled(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel.filter(*downsampled);
    return downsampled;
}

CloudPtr PointCloudRegistrar::transformCloud(CloudPtr cloud, const Eigen::Vector3f& translation) {
    CloudPtr transformed(new Cloud);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0,3) = translation.x();
    transform(1,3) = translation.y();
    transform(2,3) = translation.z();
    pcl::transformPointCloud(*cloud, *transformed, transform);
    return transformed;
}

bool PointCloudRegistrar::icpAlign(CloudPtr source, CloudPtr target, Eigen::Matrix4f& transform) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(icp_max_iterations_);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(icp_fitness_threshold_);
    icp.setMaxCorrespondenceDistance(0.05f);
    
    CloudPtr aligned(new Cloud);
    icp.align(*aligned);
    
    if (icp.hasConverged()) {
        transform = icp.getFinalTransformation();
        std::cout << "[Registrar] ICP收敛，fitness: " << icp.getFitnessScore() << std::endl;
        return true;
    }
    
    std::cerr << "[Registrar] ICP未收敛" << std::endl;
    return false;
}

CloudPtr PointCloudRegistrar::mergeClouds(const std::vector<CloudPtr>& clouds) {
    CloudPtr merged(new Cloud);
    for (const auto& cloud : clouds) {
        *merged += *cloud;
    }
    return downsample(merged);
}

// 这里是v1版本，认为点云已经在全局坐标系下，相机位姿也是全局坐标系，直接拼接，这是我生成的数据的格式，后续如果有需要再进行ICP配准，目前先不做任何变换，直接使用原始点云和位姿，后续根据实际雷达得到数据选取，确定好和硬件雷达的接口即可
bool PointCloudRegistrar::registerPointClouds() {
    if (frames_.empty()) {
        std::cerr << "[Registrar] 没有点云数据" << std::endl;
        return false;
    }
    
    std::cout << "[Registrar] 点云已在全局坐标系中，直接合并..." << std::endl;
    
    std::vector<CloudPtr> transformed_clouds;
    camera_poses_.clear();
    
    for (const auto& frame : frames_) {
        // ✅ 不做任何变换，直接使用原始点云
        transformed_clouds.push_back(frame.cloud);
        
        // ✅ 直接使用原始位姿
        camera_poses_.push_back(frame.pose);
        
        std::cout << "[Registrar] 帧 " << frame.id << ": 点数=" << frame.cloud->points.size()
                  << ", 位姿=(" << frame.pose.x() << ", " << frame.pose.y() << ", " << frame.pose.z() << ")" << std::endl;
    }
    
    // 合并所有点云
    global_cloud_ = mergeClouds(transformed_clouds);
    
    std::cout << "[Registrar] 合并完成，全局点云: " << global_cloud_->points.size() << " 点" << std::endl;
    std::cout << "[Registrar] 相机位姿数量: " << camera_poses_.size() << std::endl;
    
    return true;
}

// 这里是v2版本可以认为点云是在相机的局部坐标系下的，且可以进行ICP配准，二选一，后面用得上
// bool PointCloudRegistrar::registerPointClouds() {
//     if (frames_.empty()) {
//         std::cerr << "[Registrar] 没有点云数据" << std::endl;
//         return false;
//     }
    
//     if (use_icp_) {
//         return registerWithICP();  // ICP配准版本
//     } else {
//         return registerWithOriginalPoses();  // 直接使用原始位姿，这里认为点云是在相机的局部坐标系下的，配合下面函数使用，根据实际雷达得到数据选取，后续确定好和硬件雷达的接口即可
//     }
// }

// bool PointCloudRegistrar::registerWithOriginalPoses() {
//     std::cout << "[Registrar] 使用原始位姿构建全局点云..." << std::endl;
    
//     std::vector<CloudPtr> transformed_clouds;
//     camera_poses_.clear();
    
//     for (const auto& frame : frames_) {
//         CloudPtr transformed;
        
//         // 直接使用位姿文件中的原始坐标进行变换
//         transformed = transformCloud(frame.cloud, frame.pose);
//         camera_poses_.push_back(frame.pose);
        
//         transformed_clouds.push_back(transformed);
//         std::cout << "[Registrar] 帧 " << frame.id << " 原始位姿: ("
//                   << frame.pose.x() << ", " << frame.pose.y() << ", " << frame.pose.z() << ")" << std::endl;
//     }
    
//     global_cloud_ = mergeClouds(transformed_clouds);
    
//     std::cout << "[Registrar] 配准完成，全局点云: " << global_cloud_->points.size() << " 点" << std::endl;
    
//     return true;
// }

// bool PointCloudRegistrar::registerWithICP() {
//     if (frames_.empty()) {
//         std::cerr << "[Registrar] 没有点云数据" << std::endl;
//         return false;
//     }
    
//     std::cout << "[Registrar] 开始构建全局点云（使用原始位姿，不进行ICP）..." << std::endl;
    
//     std::vector<CloudPtr> transformed_clouds;
//     camera_poses_.clear();
    
//     for (const auto& frame : frames_) {
//         CloudPtr transformed;
        
//         // ✅ 直接使用位姿文件中的原始坐标进行变换
//         // 注意：frame.pose 已经从 {i}_frame.txt 中读取
//         transformed = transformCloud(frame.cloud, frame.pose);
//         camera_poses_.push_back(frame.pose);
        
//         transformed_clouds.push_back(transformed);
//         std::cout << "[Registrar] 帧 " << frame.id << " 使用原始位姿: ("
//                   << frame.pose.x() << ", " << frame.pose.y() << ", " << frame.pose.z() << ")" << std::endl;
//     }
    
//     // 合并所有变换后的点云
//     global_cloud_ = mergeClouds(transformed_clouds);
    
//     std::cout << "[Registrar] 配准完成，全局点云: " << global_cloud_->points.size() << " 点" << std::endl;
//     std::cout << "[Registrar] 相机位姿数量: " << camera_poses_.size() << std::endl;
    
//     return true;
// }

bool PointCloudRegistrar::saveGlobalCloud(const std::string& filename) {
    if (!global_cloud_ || global_cloud_->points.empty()) {
        std::cerr << "[Registrar] 没有全局点云可保存" << std::endl;
        return false;
    }
    
    pcl::io::savePCDFileBinary(filename, *global_cloud_);
    std::cout << "[Registrar] 保存全局点云: " << filename << std::endl;
    return true;
}

bool PointCloudRegistrar::saveCameraPoses(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Registrar] 无法保存相机位姿: " << filename << std::endl;
        return false;
    }
    
    for (const auto& pose : camera_poses_) {
        file << pose.x() << "," << pose.y() << "," << pose.z() << std::endl;
    }
    
    file.close();
    std::cout << "[Registrar] 保存相机位姿: " << filename << std::endl;
    return true;
}
