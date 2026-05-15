// pointcloud_registrar.hpp
#ifndef POINTCLOUD_REGISTRAR_HPP
#define POINTCLOUD_REGISTRAR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr;

struct FrameData {
    CloudPtr cloud;
    Eigen::Vector3f pose;  // 全局坐标 (x, y, z)
    int id;
};

class PointCloudRegistrar {
public:
    PointCloudRegistrar();
    ~PointCloudRegistrar();
    
    // 加载数据
    bool loadFrames(const std::string& folder_path, bool has_pose_files = true);
    
    // 配准点云
    bool registerPointClouds();
    bool registerWithOriginalPoses();
    bool registerWithICP();
    
    // 获取结果
    CloudPtr getGlobalCloud() const { return global_cloud_; }
    std::vector<Eigen::Vector3f> getCameraPoses() const { return camera_poses_; }
    std::vector<FrameData> getFrames() const { return frames_; }
    
    // 保存结果
    bool saveGlobalCloud(const std::string& filename);
    bool saveCameraPoses(const std::string& filename);
    
    // 参数设置
    void setVoxelSize(float size) { voxel_size_ = size; }
    void setIcpMaxIterations(int iter) { icp_max_iterations_ = iter; }
    void setIcpFitnessThreshold(float thresh) { icp_fitness_threshold_ = thresh; }
    void setUseICP(bool use) { use_icp_ = use; }
    
private:
    // ICP配准
    bool icpAlign(CloudPtr source, CloudPtr target, Eigen::Matrix4f& transform);
    
    // 通过位姿变换点云
    CloudPtr transformCloud(CloudPtr cloud, const Eigen::Vector3f& translation);
    
    // 合并点云
    CloudPtr mergeClouds(const std::vector<CloudPtr>& clouds);
    
    // 降采样
    CloudPtr downsample(CloudPtr cloud);
    
    std::vector<FrameData> frames_;
    std::vector<Eigen::Vector3f> camera_poses_;
    CloudPtr global_cloud_;
    
    // 参数
    float voxel_size_ = 0.003f;
    int icp_max_iterations_ = 50;
    float icp_fitness_threshold_ = 0.0001f;
    bool use_icp_ = false;  // 默认不使用ICP
};

#endif
