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

};

class PlaneExtractor {
public:
    PlaneExtractor();
    ~PlaneExtractor();
    
    std::vector<FinitePlane> extractPlanes(CloudPtr cloud);
    
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
    
    // 参数设置
    void setDistanceThreshold(float thresh) { distance_threshold_ = thresh; }
    void setMinPlanePoints(int min_pts) { min_plane_points_ = min_pts; }
    void setMaxIterations(int iter) { max_iterations_ = iter; }
    void setVoxelSize(float size) { voxel_size_ = size; }
    
private:
    CloudPtr downsample(CloudPtr cloud);
    void computePlaneBounds(FinitePlane& plane);
    void segmentConnectedComponents(FinitePlane& plane);
    // bool isPointInPlaneBounds(const FinitePlane& plane, const Eigen::Vector3f& point, float margin = 0.01f);

    // ✅ 添加这两个函数声明
    void computePlaneLocalFrame(FinitePlane& plane);
    void computePlaneConvexHull(FinitePlane& plane);
    
    float distance_threshold_ = 0.005f;
    int min_plane_points_ = 500;
    int max_iterations_ = 1000;
    float voxel_size_ = 0.003f;
};

#endif
