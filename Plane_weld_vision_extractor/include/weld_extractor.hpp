// weld_extractor.hpp
#ifndef WELD_EXTRACTOR_HPP
#define WELD_EXTRACTOR_HPP

#include "pointcloud_registrar.hpp"
#include "plane_extractor.hpp"
#include "corner_classifier.hpp"
#include <string>

class WeldExtractor {
public:
    WeldExtractor();
    ~WeldExtractor();
    
    // ========== 步骤1：点云配准 ==========
    bool loadFrames(const std::string& folder_path, bool has_pose_files = true);
    bool registerPointClouds();
    CloudPtr getGlobalCloud() const { return registrar_.getGlobalCloud(); }
    std::vector<Eigen::Vector3f> getCameraPoses() const { return registrar_.getCameraPoses(); }
    
    // ========== 步骤2：焊缝提取 ==========
    std::vector<WeldSeam> extractWeldSeams(CloudPtr cloud, 
                                            const std::vector<Eigen::Vector3f>& camera_poses);
    
    // ========== 一步到位 ==========
    bool process(const std::string& dataset_folder, 
                 const std::string& output_folder,
                 bool has_pose_files = true);
    
    // ========== 保存结果 ==========
    bool saveGlobalCloud(const std::string& filename);
    bool saveWeldSeams(const std::string& filename);
    // bool saveWeldSeams(const std::vector<WeldSeam>& seams, const std::string& filename);
    bool saveCameraPoses(const std::string& filename);
    
    // ========== 参数设置 ==========
    void setVoxelSize(float size) { registrar_.setVoxelSize(size); plane_extractor_.setVoxelSize(size); }
    void setPlaneThreshold(float thresh) { plane_extractor_.setDistanceThreshold(thresh); }
    void setMinPlanePoints(int pts) { plane_extractor_.setMinPlanePoints(pts); }
    void setPathSpacing(float spacing) { classifier_.setPathSpacing(spacing); }
    void setWeldAngle(float angle) { classifier_.setWeldAngle(angle); }

    // 模式设置
    void setExtractionMode(ExtractionMode mode) { classifier_.setExtractionMode(mode); }
    void setMinWeldLength(float length) { classifier_.setMinWeldLength(length); }
    void setUseICP(bool use) { registrar_.setUseICP(use); }

private:
    PointCloudRegistrar registrar_;
    PlaneExtractor plane_extractor_;
    CornerClassifier classifier_;
    
    std::vector<WeldSeam> seams_;
    std::string output_folder_;
};

#endif
