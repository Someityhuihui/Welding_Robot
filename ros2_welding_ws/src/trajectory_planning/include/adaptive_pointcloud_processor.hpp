// adaptive_pointcloud_processor.hpp
#ifndef ADAPTIVE_POINTCLOUD_PROCESSOR_HPP
#define ADAPTIVE_POINTCLOUD_PROCESSOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef Cloud::Ptr CloudPtr;


struct AdaptiveConfig {
    // 目标点云规模 - 改为论文标准
    size_t target_points_min = 50000;   // 修改：40000点
    size_t target_points_max = 100000;   // 修改：50000点（论文标准）
    
    // 体素大小范围
    float voxel_size_min = 0.0015f;     // 修改：1.5mm（更高精度）
    float voxel_size_max = 0.005f;      // 修改：5mm
    
    // 关键帧判断阈值
    float bbox_change_ratio = 0.15f;
    float min_overlap_ratio = 0.30f;
    
    // 增量拼接阈值 - 论文标准
    float icp_fitness_threshold = 0.0001f;  // 修改：更严格
    int max_icp_iterations = 50;            // 修改：30次（论文标准）
    float max_correspondence_distance = 0.02f;  // 修改：2cm
    int ransac_iterations = 200;            // 新增：RANSAC迭代次数
};

class AdaptivePointCloudProcessor {
public:
    AdaptivePointCloudProcessor();
    ~AdaptivePointCloudProcessor();
    
    // 配置接口
    void setConfig(const AdaptiveConfig& config);
    void setUseKeyFrameStrategy(bool use) { use_keyframe_strategy_ = use; }
    void setUseIncrementalStitching(bool use) { use_incremental_stitching_ = use; }
    
    // 核心功能
    CloudPtr adaptiveDownsample(CloudPtr cloud, int depth = 0);
    bool isKeyFrame(CloudPtr cloud);
    bool incrementalStitch(CloudPtr new_frame, Eigen::Matrix4f& transform);
    CloudPtr getGlobalCloud() const { return global_cloud_; }
    int getProcessedFrames() const { return processed_frames_; }
    int getSkippedFrames() const { return skipped_frames_; }
    
    // 辅助功能
    void reset();
    void computeBoundingBox(CloudPtr cloud, Eigen::Vector3f& min_pt, Eigen::Vector3f& max_pt);
    float computeOverlapRatio(CloudPtr cloud1, CloudPtr cloud2);
    float computeBBoxChangeRatio(CloudPtr cloud1, CloudPtr cloud2);

    // 新增：获取当前配置
    const AdaptiveConfig& getConfig() const { return config_; }
    // 新增：设置迭代次数
    void setMaxICPIterations(int iterations) { config_.max_icp_iterations = iterations; }
    
private:
    AdaptiveConfig config_;
    CloudPtr global_cloud_;           // 全局拼接点云
    CloudPtr prev_keyframe_;          // 上一关键帧
    Eigen::Vector3f prev_bbox_min_;   // 上一包围盒最小值
    Eigen::Vector3f prev_bbox_max_;   // 上一包围盒最大值
    
    bool use_incremental_stitching_;
    bool use_keyframe_strategy_;
    int processed_frames_;
    int skipped_frames_;
    
    float calculateOptimalVoxelSize(CloudPtr cloud);
    float computeAverageDensity(CloudPtr cloud);
    bool fastICP(CloudPtr source, CloudPtr target, Eigen::Matrix4f& transform, CloudPtr& aligned);
    CloudPtr fuseAndDownsample(CloudPtr cloud1, CloudPtr cloud2, const Eigen::Matrix4f& transform);
    void filterOutliers(CloudPtr cloud);
    bool validateStitchResult(CloudPtr aligned, CloudPtr target, const Eigen::Matrix4f& transform);
};

#endif
