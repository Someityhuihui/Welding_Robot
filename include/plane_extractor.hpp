// plane_extractor.hpp - 添加成员变量和函数声明

#ifndef PLANE_EXTRACTOR_HPP
#define PLANE_EXTRACTOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr;

struct FinitePlane {
    int id;
    Eigen::Vector3f normal;
    Eigen::Vector3f center;
    float d;
    CloudPtr cloud;
    int point_count;
    
    // 连通分量信息
    std::vector<CloudPtr> components;
    std::vector<Eigen::Vector3f> component_centers;
    std::vector<Eigen::Vector3f> component_bbox_min;
    std::vector<Eigen::Vector3f> component_bbox_max;
    
    // 边界
    float min_x, max_x, min_y, max_y, min_z, max_z;
    
    // 可视化颜色
    int r, g, b;

    // ✅ 添加：用于精确边界判断的成员
    Eigen::Vector3f local_x;                     // 平面内X轴
    Eigen::Vector3f local_y;                    // 平面内Y轴
    Eigen::Vector3f origin;                     // 局部坐标系原点
    std::vector<Eigen::Vector2f> hull_2d;       // 2D凸包顶点

    // 新增：3σ补全相关
    float distance_mean = 0.0f;      // 距离均值
    float distance_stddev = 0.0f;    // 距离标准差
    float sigma_threshold = 0.0f;    // 3σ阈值

};


class PlaneExtractor {
public:
    PlaneExtractor();
    ~PlaneExtractor();
    
    std::vector<FinitePlane> extractPlanes(CloudPtr cloud);

    std::vector<FinitePlane> mergeSimilarPlanes(const std::vector<FinitePlane>& planes);
    
    bool computeTriplePlaneIntersection(const FinitePlane& p1,
                                         const FinitePlane& p2,
                                         const FinitePlane& p3,
                                         Eigen::Vector3f& intersection);
    
    bool computePlaneIntersectionLine(const FinitePlane& p1,
                                       const FinitePlane& p2,
                                       Eigen::Vector3f& direction,
                                       Eigen::Vector3f& point_on_line);

    // 精确判断点是否在平面边界内（使用2D凸包）
    bool isPointInPlaneBoundsExact(const FinitePlane& plane, 
                                    const Eigen::Vector3f& point, 
                                    float margin = 0.01f);

    // 简单边界判断（使用AABB）
    bool isPointInPlaneBounds(const FinitePlane& plane, 
                               const Eigen::Vector3f& point,
                               float margin = 0.01f);
    
    CloudPtr sampleIntersectionLine(const FinitePlane& p1,
                                     const FinitePlane& p2,
                                     float step = 0.005f);
    
    void orientNormalsOutward(std::vector<FinitePlane>& planes,
                              const std::vector<Eigen::Vector3f>& camera_poses);

    // 区域生长法提取平面
    std::vector<FinitePlane> extractPlanesRegionGrowing(CloudPtr cloud);
    
    // 法向量估计（用于区域生长）
    void estimateNormals(CloudPtr cloud, 
                         pcl::PointCloud<pcl::Normal>::Ptr normals,
                         float radius = 0.03f);

    // 统一的平面后处理函数：连通分量分割 + 索引修正
    std::vector<FinitePlane> postProcessPlanes(const std::vector<FinitePlane>& input_planes);
    
    // 参数设置
    void setDistanceThreshold(float thresh) { distance_threshold_ = thresh; }
    void setMinPlanePoints(int min_pts) { min_plane_points_ = min_pts; }
    void setMaxIterations(int iter) { max_iterations_ = iter; }
    void setVoxelSize(float size) { voxel_size_ = size; }
    // 区域生长参数设置
    void setNormalSmoothnessThreshold(float threshold) { normal_smoothness_threshold_ = threshold; }
    void setCurvatureThreshold(float threshold) { curvature_threshold_ = threshold; }
    void setMinClusterSize(int size) { min_cluster_size_ = size; }

    // 新增：3σ距离补全
    void completePlanesWithSigma(std::vector<FinitePlane>& planes, CloudPtr cloud);
    
    // 新增：精确平面交线计算（带边界裁剪）
    CloudPtr computePreciseIntersectionLine(const FinitePlane& p1, const FinitePlane& p2);

    
private:
    CloudPtr downsample(CloudPtr cloud);
    void computePlaneBounds(FinitePlane& plane);
    void segmentConnectedComponents(FinitePlane& plane);
    // bool isPointInPlaneBounds(const FinitePlane& plane, const Eigen::Vector3f& point, float margin = 0.01f);

    float normal_smoothness_threshold_ = 30.0f;  // 法向量夹角阈值（度）
    float curvature_threshold_ = 0.2f;          // 曲率阈值
    int min_cluster_size_ = 1000;                 // 最小聚类点数

    float sigma_threshold_ = 3.0f;  // 3σ阈值

    // ✅ 添加这两个函数声明
    void computePlaneLocalFrame(FinitePlane& plane);
    void computePlaneConvexHull(FinitePlane& plane);
    
    float distance_threshold_ = 0.1f;
    int min_plane_points_ = 500;
    int max_iterations_ = 1000;
    float voxel_size_ = 0.003f;

    const std::vector<std::tuple<int, int, int>> colors_ = {
        {255, 0, 0},     // 红色
        {0, 255, 0},     // 绿色
        {0, 0, 255},     // 蓝色
        {255, 255, 0},   // 黄色
        {255, 0, 255},   // 品红
        {0, 255, 255},   // 青色
        {255, 128, 0},   // 橙色
        {128, 0, 255},   // 紫色
        {255, 0, 128},   // 粉红
        {0, 128, 255},   // 天蓝
        {128, 255, 0},   // 黄绿
        {255, 128, 128}, // 浅红
        {128, 255, 128}, // 浅绿
        {128, 128, 255}, // 浅蓝
        {255, 255, 128}  // 浅黄
    };
};

#endif
