// adaptive_pointcloud_processor.cpp
#include "adaptive_pointcloud_processor.hpp"
#include <random>

AdaptivePointCloudProcessor::AdaptivePointCloudProcessor()
    : global_cloud_(new Cloud)
    , prev_keyframe_(new Cloud)
    , use_incremental_stitching_(true)
    , use_keyframe_strategy_(true)
    , processed_frames_(0)
    , skipped_frames_(0) {
    
    prev_bbox_min_ = Eigen::Vector3f(0, 0, 0);
    prev_bbox_max_ = Eigen::Vector3f(0, 0, 0);
}


// ✅ 添加这个函数
void AdaptivePointCloudProcessor::setConfig(const AdaptiveConfig& config) {
    config_ = config;
    
    std::cout << "[AdaptiveProcessor] Config updated: "
              << "target_points=[" << config_.target_points_min << "," << config_.target_points_max << "], "
              << "voxel_size=[" << config_.voxel_size_min << "," << config_.voxel_size_max << "], "
              << "icp_iterations=" << config_.max_icp_iterations
              << std::endl;
}

AdaptivePointCloudProcessor::~AdaptivePointCloudProcessor() {}

// adaptive_pointcloud_processor.cpp

float AdaptivePointCloudProcessor::calculateOptimalVoxelSize(CloudPtr cloud) {
    if (cloud->points.empty()) return config_.voxel_size_max;  // 改用最大值
    
    size_t current_size = cloud->points.size();
    
    // ✅ 使用配置中的目标值，而不是硬编码
    size_t target_size = config_.target_points_max;
    
    // 如果点数已经在目标范围内，返回当前体素大小（不继续缩小）
    if (current_size <= target_size * 1.2f) {
        // 返回一个较大的值，避免继续降采样
        return config_.voxel_size_max;
    }
    
    // 计算需要的体素大小
    float ratio = std::cbrt((float)current_size / target_size);
    float voxel_size = config_.voxel_size_min * ratio;
    
    // 限制范围
    voxel_size = std::clamp(voxel_size, config_.voxel_size_min, config_.voxel_size_max);
    
    // ✅ 如果体素大小已经接近最小值，直接返回最大值（停止降采样）
    if (voxel_size < config_.voxel_size_min * 1.1f) {
        std::cout << "[AdaptiveDownsample] Voxel size at minimum, stopping further downsampling" << std::endl;
        return config_.voxel_size_max;
    }
    
    return voxel_size;
}

// adaptive_pointcloud_processor.cpp

CloudPtr AdaptivePointCloudProcessor::adaptiveDownsample(CloudPtr cloud, int depth) {
    if (!cloud || cloud->points.empty()) {
        return CloudPtr(new Cloud);
    }
    
    // 递归深度限制
    const int MAX_RECURSION_DEPTH = 5;
    if (depth >= MAX_RECURSION_DEPTH) {
        std::cout << "[AdaptiveDownsample] Max recursion depth (" << MAX_RECURSION_DEPTH 
                  << ") reached, stopping." << std::endl;
        return cloud;
    }
    
    size_t original_size = cloud->points.size();
    
    // ✅ 如果点数已经在目标范围内，直接返回
    if (original_size <= config_.target_points_max * 1.2f) {
        std::cout << "[AdaptiveDownsample] Point count within target range (" 
                  << original_size << " <= " << config_.target_points_max * 1.2f 
                  << "), skipping downsampling." << std::endl;
        return cloud;
    }
    
    float optimal_voxel = calculateOptimalVoxelSize(cloud);
    
    // ✅ 如果体素大小太大，无法有效降采样
    if (optimal_voxel >= config_.voxel_size_max * 0.9f) {
        std::cout << "[AdaptiveDownsample] Voxel size too large (" << optimal_voxel 
                  << "m), cannot downsample effectively. Stopping." << std::endl;
        return cloud;
    }
    
    std::cout << "[AdaptiveDownsample] Original: " << original_size 
              << " points, target: " << config_.target_points_max
              << ", voxel: " << optimal_voxel << "m" << std::endl;
    
    CloudPtr downsampled(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(optimal_voxel, optimal_voxel, optimal_voxel);
    voxel.filter(*downsampled);
    
    size_t new_size = downsampled->points.size();
    float reduction_ratio = (float)new_size / original_size;
    
    std::cout << "[AdaptiveDownsample] Result: " << new_size 
              << " points (ratio: " << reduction_ratio << ")" << std::endl;
    
    // ✅ 如果降采样效果不明显（减少少于10%），停止递归
    if (reduction_ratio > 0.9f) {
        std::cout << "[AdaptiveDownsample] Downsampling ineffective (<10% reduction), stopping." << std::endl;
        return downsampled;
    }
    
    // ✅ 如果点数仍然太多，且还有递归深度，继续
    if (new_size > config_.target_points_max * 1.2f && depth + 1 < MAX_RECURSION_DEPTH) {
        std::cout << "[AdaptiveDownsample] Still too many points (" << new_size 
                  << "), recursing (depth " << depth + 1 << ")" << std::endl;
        return adaptiveDownsample(downsampled, depth + 1);
    }
    
    // 统计滤波去噪
    filterOutliers(downsampled);
    
    return downsampled;
}


void AdaptivePointCloudProcessor::filterOutliers(CloudPtr cloud) {
    if (cloud->points.size() < 100) return;
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    
    CloudPtr filtered(new Cloud);
    sor.filter(*filtered);
    
    if (!filtered->points.empty()) {
        cloud->swap(*filtered);
    }
}

void AdaptivePointCloudProcessor::computeBoundingBox(CloudPtr cloud, 
                                                      Eigen::Vector3f& min_pt, 
                                                      Eigen::Vector3f& max_pt) {
    pcl::PointXYZ min_pt_pcl, max_pt_pcl;
    pcl::getMinMax3D(*cloud, min_pt_pcl, max_pt_pcl);
    
    min_pt = Eigen::Vector3f(min_pt_pcl.x, min_pt_pcl.y, min_pt_pcl.z);
    max_pt = Eigen::Vector3f(max_pt_pcl.x, max_pt_pcl.y, max_pt_pcl.z);
}

float AdaptivePointCloudProcessor::computeBBoxChangeRatio(CloudPtr cloud1, CloudPtr cloud2) {
    Eigen::Vector3f min1, max1, min2, max2;
    computeBoundingBox(cloud1, min1, max1);
    computeBoundingBox(cloud2, min2, max2);
    
    Eigen::Vector3f size1 = max1 - min1;
    Eigen::Vector3f size2 = max2 - min2;
    
    float volume1 = size1.x() * size1.y() * size1.z();
    float volume2 = size2.x() * size2.y() * size2.z();
    
    if (volume1 < 1e-9) return 0.0f;
    
    return std::abs(volume2 - volume1) / volume1;
}

float AdaptivePointCloudProcessor::computeOverlapRatio(CloudPtr cloud1, CloudPtr cloud2) {
    if (cloud1->points.empty() || cloud2->points.empty()) return 0.0f;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud1);
    
    int sample_size = std::min(5000, (int)cloud2->points.size());
    int overlap_count = 0;
    float search_radius = 0.01f;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, cloud2->points.size() - 1);
    
    for (int i = 0; i < sample_size; i++) {
        int idx = dis(gen);
        std::vector<int> indices;
        std::vector<float> distances;
        if (kdtree.radiusSearch(cloud2->points[idx], search_radius, indices, distances) > 0) {
            overlap_count++;
        }
    }
    
    return (float)overlap_count / sample_size;
}

bool AdaptivePointCloudProcessor::isKeyFrame(CloudPtr cloud) {
    if (!use_keyframe_strategy_) return true;
    
    // 第一帧总是关键帧
    if (prev_keyframe_->points.empty()) {
        pcl::copyPointCloud(*cloud, *prev_keyframe_);
        computeBoundingBox(cloud, prev_bbox_min_, prev_bbox_max_);
        std::cout << "[KeyFrame] First frame accepted as keyframe" << std::endl;
        return true;
    }
    
    // 计算变化指标
    float bbox_change = computeBBoxChangeRatio(prev_keyframe_, cloud);
    float overlap = computeOverlapRatio(prev_keyframe_, cloud);
    
    bool is_keyframe = (bbox_change > config_.bbox_change_ratio || 
                        overlap < config_.min_overlap_ratio);
    
    if (is_keyframe) {
        std::cout << "[KeyFrame] New keyframe: bbox_change=" << bbox_change 
                  << ", overlap=" << overlap << std::endl;
        pcl::copyPointCloud(*cloud, *prev_keyframe_);
        computeBoundingBox(cloud, prev_bbox_min_, prev_bbox_max_);
        processed_frames_++;
    } else {
        std::cout << "[KeyFrame] Skipping frame: bbox_change=" << bbox_change 
                  << ", overlap=" << overlap << std::endl;
        skipped_frames_++;
    }
    
    return is_keyframe;
}

bool AdaptivePointCloudProcessor::fastICP(CloudPtr source, CloudPtr target, 
                                           Eigen::Matrix4f& transform, CloudPtr& aligned) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    
    // 论文标准参数
    icp.setMaximumIterations(config_.max_icp_iterations);  // 30次
    icp.setTransformationEpsilon(1e-10);                   // 更严格
    icp.setEuclideanFitnessEpsilon(config_.icp_fitness_threshold);
    icp.setMaxCorrespondenceDistance(config_.max_correspondence_distance);
    icp.setRANSACIterations(config_.ransac_iterations);    // 200次
    
    // 多点云配准时的重叠率检查
    icp.setRANSACOutlierRejectionThreshold(0.05);
    
    aligned.reset(new Cloud);
    icp.align(*aligned);
    
    if (icp.hasConverged()) {
        transform = icp.getFinalTransformation();
        std::cout << "[ICP] Converged with fitness: " << icp.getFitnessScore();
        return true;
    } else {
        std::cout << "[ICP] Did not converge!" << std::endl;
        return false;
    }
}


bool AdaptivePointCloudProcessor::validateStitchResult(CloudPtr aligned, CloudPtr target, 
                                                        const Eigen::Matrix4f& transform) {
    float overlap = computeOverlapRatio(aligned, target);
    if (overlap < 0.2f) {
        std::cout << "[Validate] Stitch validation failed: overlap=" << overlap << std::endl;
        return false;
    }
    
    // 检查变换是否合理（平移不能太大）
    float translation = transform.block<3,1>(0,3).norm();
    if (translation > 0.5f) {
        std::cout << "[Validate] Translation too large: " << translation << "m" << std::endl;
        return false;
    }
    
    return true;
}

CloudPtr AdaptivePointCloudProcessor::fuseAndDownsample(CloudPtr cloud1, CloudPtr cloud2, 
                                                         const Eigen::Matrix4f& transform) {
    // 变换第二片点云
    CloudPtr transformed_cloud2(new Cloud);
    pcl::transformPointCloud(*cloud2, *transformed_cloud2, transform);
    
    // 合并点云
    CloudPtr fused(new Cloud);
    *fused += *cloud1;
    *fused += *transformed_cloud2;
    
    // 降采样合并后的点云
    return adaptiveDownsample(fused);
}

bool AdaptivePointCloudProcessor::incrementalStitch(CloudPtr new_frame, 
                                                     Eigen::Matrix4f& transform) {
    if (!use_incremental_stitching_) return false;
    
    // 如果没有全局点云，直接使用新帧
    if (global_cloud_->points.empty()) {
        global_cloud_ = adaptiveDownsample(new_frame);
        std::cout << "[Stitch] Initialized global cloud with " 
                  << global_cloud_->points.size() << " points" << std::endl;
        transform = Eigen::Matrix4f::Identity();
        processed_frames_++;
        return true;
    }
    
    // 快速配准
    CloudPtr aligned;
    if (!fastICP(new_frame, global_cloud_, transform, aligned)) {
        std::cout << "[Stitch] ICP failed for this frame" << std::endl;
        return false;
    }
    
    // 验证配准质量
    if (!validateStitchResult(aligned, global_cloud_, transform)) {
        std::cout << "[Stitch] Stitch validation failed" << std::endl;
        return false;
    }
    
    // 融合点云
    CloudPtr new_global = fuseAndDownsample(global_cloud_, new_frame, transform);
    
    global_cloud_ = new_global;
    processed_frames_++;
    
    std::cout << "[Stitch] Success. Global cloud now has " 
              << global_cloud_->points.size() << " points" << std::endl;
    
    return true;
}

void AdaptivePointCloudProcessor::reset() {
    global_cloud_->clear();
    prev_keyframe_->clear();
    processed_frames_ = 0;
    skipped_frames_ = 0;
    prev_bbox_min_ = Eigen::Vector3f(0, 0, 0);
    prev_bbox_max_ = Eigen::Vector3f(0, 0, 0);
}
