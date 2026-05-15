// corner_classifier.hpp
#ifndef CORNER_CLASSIFIER_HPP
#define CORNER_CLASSIFIER_HPP

#include <Eigen/Dense>
#include <vector>
#include "plane_extractor.hpp"

// 角点结构体
struct CornerPoint {
    Eigen::Vector3f position;
    std::vector<int> plane_ids;
    float confidence;
    bool is_concave;                    // 内凹角（需焊接）
    Eigen::Vector3f inward_direction;   // 指向模型内部的方向
};

// 焊缝结构体
struct WeldSeam {
    int id;
    int plane_id1, plane_id2;           // 组成焊缝的两个平面
    std::vector<Eigen::Vector3f> path;  // 路径点
    std::vector<Eigen::Vector3f> normals; // 法向量
    std::vector<Eigen::Quaternionf> orientations; // 焊枪姿态
    float length;                       // 焊缝长度
    bool is_corner_weld;                // 是否为角焊缝（三平面交线）
};

// 添加枚举类型
enum ExtractionMode {
    MODE_CORNER_ONLY = 0,    // 边角模式：只提取三平面相交的三条焊缝
    MODE_LONG_ONLY = 1,      // 长条模式：只提取两平面相交的长焊缝
    MODE_BOTH = 2            // 混合模式：两者都提取
};

class CornerClassifier {
public:
    CornerClassifier();
    ~CornerClassifier();
    
    // 设置点云（用于内外判断）
    void setPointCloud(CloudPtr cloud);
    
    // 设置相机位姿（用于辅助判断）
    void setCameraPoses(const std::vector<Eigen::Vector3f>& poses);
    
    // 分类所有角点
    // Eigen::Vector3f getOutwardNormal(const FinitePlane& plane, const Eigen::Vector3f& corner_point);
    // bool isConcaveCornerGeneral(const CornerPoint& corner, const std::vector<FinitePlane>& planes);
    std::vector<CornerPoint> classifyCorners(
        const std::vector<FinitePlane>& planes,
        const std::vector<Eigen::Vector3f>& intersection_points,
        const std::vector<std::tuple<int, int, int>>& intersection_plane_indices);  // ✅ 添加参数
    
    // 提取所有焊缝（角焊缝 + 长焊缝）
    std::vector<WeldSeam> extractWeldSeams(const std::vector<FinitePlane>& planes,
                                            const std::vector<CornerPoint>& corners);

    
    // 导出CSV
    bool exportWeldSeamsToCSV(const std::vector<WeldSeam>& seams, const std::string& filename);
    
    // 参数设置
    void setPathSpacing(float spacing) { path_spacing_ = spacing; }
    void setWeldAngle(float angle) { weld_angle_ = angle; }
    void setOutwardThreshold(float thresh) { outward_threshold_ = thresh; }

    // 设置提取模式
    void setExtractionMode(ExtractionMode mode) { extraction_mode_ = mode; }
    ExtractionMode getExtractionMode() const { return extraction_mode_; }
    
    // 设置最小焊缝长度（过滤短焊缝，单位：米）
    void setMinWeldLength(float length) { min_weld_length_ = length; }

    // 新增：判断相机是否可以看到角点
    bool isCameraVisibleToCorner(const CornerPoint& corner, 
                                  const std::vector<FinitePlane>& planes,
                                  const Eigen::Vector3f& camera_pos);
    
    // 新增：计算加权分数（基于可见相机）
    float computeCornerScoreFromVisibleCameras(const CornerPoint& corner,
                                                const std::vector<FinitePlane>& planes,
                                                const std::vector<Eigen::Vector3f>& camera_poses);
    
    // 新增：保存角点信息到文件
    void saveCornersToFile(const std::vector<CornerPoint>& corners, const std::string& filename);
    
    
private:
    // 内凹角判断方法
    bool classifyByVectorSum(const CornerPoint& corner, const std::vector<FinitePlane>& planes);
    bool classifyByCameraPose(const CornerPoint& corner, const std::vector<FinitePlane>& planes);
    bool classifyByNormalDistribution(const CornerPoint& corner, const std::vector<FinitePlane>& planes);
    
    // 辅助函数
    bool isDirectionOutward(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction);
    float computeAngle(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2);
    Eigen::Vector3f vectorToPlaneCenter(const CornerPoint& corner, const FinitePlane& plane);
    
    // 焊缝生成
    WeldSeam createWeldFromIntersection(const FinitePlane& p1, const FinitePlane& p2, int id);
    void samplePathOnLine(CloudPtr line_points, WeldSeam& seam);
    
    CloudPtr cloud_;
    std::vector<Eigen::Vector3f> camera_poses_;
    Eigen::Vector3f cloud_center_;
    Eigen::Vector3f cloud_bbox_min_, cloud_bbox_max_;
    
    // 参数
    float path_spacing_ = 0.005f;    // 5mm
    float weld_angle_ = 45.0f;       // 焊枪角度
    float outward_threshold_ = 0.3f; // 外部判定阈值
    ExtractionMode extraction_mode_ = MODE_CORNER_ONLY;  // 默认为边角模式
    float min_weld_length_ = 0.01f;  // 最小焊缝长度 50mm
    

    // 辅助函数
    bool isPointInCloudBounds(const Eigen::Vector3f& point, float margin = 0.02f);
    void filterWeldSeamByBounds(WeldSeam& seam);

    // 辅助函数：检查点是否在平行六面体内
    bool isPointInParallelepiped(const Eigen::Vector3f& point,
                                  const CornerPoint& corner,
                                  const std::vector<FinitePlane>& planes);

    Eigen::Vector3f getValidPointInPlane(const FinitePlane& plane, 
                                          const Eigen::Vector3f& corner_point,
                                          const Eigen::Vector3f& other_dir1,
                                          const Eigen::Vector3f& other_dir2);
};

#endif
