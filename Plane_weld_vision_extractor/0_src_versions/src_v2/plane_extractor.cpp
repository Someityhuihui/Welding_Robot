// plane_extractor.cpp
#include "plane_extractor.hpp"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <cfloat>
#include <algorithm>
#include <tuple>

PlaneExtractor::PlaneExtractor() {}

PlaneExtractor::~PlaneExtractor() {}

CloudPtr PlaneExtractor::downsample(CloudPtr cloud) {
    CloudPtr downsampled(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel.filter(*downsampled);
    return downsampled;
}

void PlaneExtractor::computePlaneBounds(FinitePlane& plane) {
    if (!plane.cloud || plane.cloud->points.empty()) {
        plane.min_x = plane.max_x = plane.min_y = plane.max_y = plane.min_z = plane.max_z = 0;
        return;
    }
    
    plane.min_x = plane.min_y = plane.min_z = FLT_MAX;
    plane.max_x = plane.max_y = plane.max_z = -FLT_MAX;
    
    for (const auto& p : plane.cloud->points) {
        if (p.x < plane.min_x) plane.min_x = p.x;
        if (p.x > plane.max_x) plane.max_x = p.x;
        if (p.y < plane.min_y) plane.min_y = p.y;
        if (p.y > plane.max_y) plane.max_y = p.y;
        if (p.z < plane.min_z) plane.min_z = p.z;
        if (p.z > plane.max_z) plane.max_z = p.z;
    }
    
    // 计算平面中心
    plane.center.setZero();
    for (const auto& p : plane.cloud->points) {
        plane.center += Eigen::Vector3f(p.x, p.y, p.z);
    }
    plane.center /= plane.cloud->points.size();
}

// ========== 连通分量分割函数 ==========
void PlaneExtractor::segmentConnectedComponents(FinitePlane& plane) {
    if (!plane.cloud || plane.cloud->points.empty()) return;
    
    // 使用欧氏聚类分割
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(plane.cloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02f);  // 2cm，根据点云密度调整
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(1000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(plane.cloud);
    ec.extract(cluster_indices);
    
    std::cout << "  [连通分量] 平面 " << plane.id << " 分割为 " << cluster_indices.size() << " 个连通分量" << std::endl;
    
    for (const auto& indices : cluster_indices) {
        CloudPtr component(new Cloud);
        for (int idx : indices.indices) {
            component->push_back(plane.cloud->points[idx]);
        }
        
        // 计算分量中心
        Eigen::Vector3f center(0, 0, 0);
        for (const auto& p : component->points) {
            center += Eigen::Vector3f(p.x, p.y, p.z);
        }
        center /= component->points.size();
        
        // 计算分量边界
        Eigen::Vector3f bbox_min(FLT_MAX, FLT_MAX, FLT_MAX);
        Eigen::Vector3f bbox_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const auto& p : component->points) {
            bbox_min = bbox_min.cwiseMin(Eigen::Vector3f(p.x, p.y, p.z));
            bbox_max = bbox_max.cwiseMax(Eigen::Vector3f(p.x, p.y, p.z));
        }
        
        plane.components.push_back(component);
        plane.component_centers.push_back(center);
        plane.component_bbox_min.push_back(bbox_min);
        plane.component_bbox_max.push_back(bbox_max);
        
        std::cout << "    分量 " << plane.components.size() - 1 
                  << ": 点数=" << component->points.size()
                  << ", 中心=(" << center.x() << "," << center.y() << "," << center.z() << ")" << std::endl;
    }
}

// ========== 核心修改：提取平面 + 连通域分割 ==========
std::vector<FinitePlane> PlaneExtractor::extractPlanes(CloudPtr cloud) {
    std::vector<FinitePlane> planes;
    
    CloudPtr downsampled = downsample(cloud);
    std::cout << "[PlaneExtractor] 降采样后点数: " << downsampled->points.size() << std::endl;
    
    // 去噪
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(downsampled);
    sor.setMeanK(20);
    sor.setStddevMulThresh(1.0);
    sor.filter(*downsampled);
    
    CloudPtr remaining(new Cloud);
    *remaining = *downsampled;
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients coeff;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold_);
    seg.setMaxIterations(max_iterations_);
    
    int max_planes = 20;  // 最多提取20个平面，避免过度分割
    
    // 预定义颜色列表
    std::vector<std::tuple<int, int, int>> colors = {
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
    
    for (int attempt = 0; attempt < max_planes; attempt++) {
        if (remaining->points.size() < (size_t)min_plane_points_) break;
        
        seg.setInputCloud(remaining);
        seg.segment(*inliers, coeff);
        
        if (inliers->indices.size() < (size_t)min_plane_points_) break;
        
        FinitePlane plane;
        plane.id = planes.size();  // 使用向量索引作为 ID
        plane.normal = Eigen::Vector3f(coeff.values[0], coeff.values[1], coeff.values[2]);
        plane.normal.normalize();
        plane.d = coeff.values[3];
        plane.point_count = inliers->indices.size();
        
        // 提取平面点云
        plane.cloud.reset(new Cloud);
        extract.setInputCloud(remaining);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane.cloud);
        extract.setNegative(true);
        extract.filter(*remaining);
        
        // 计算边界
        computePlaneBounds(plane);
        
        // ===== 关键：分割连通分量 =====
        segmentConnectedComponents(plane);
        
        // 如果分割出多个分量，为每个分量创建单独的平面记录
        if (plane.components.size() > 1) {
            std::cout << "  [分裂] 平面 " << plane.id << " 分裂为 " << plane.components.size() << " 个独立平面" << std::endl;
            
            for (size_t comp_idx = 0; comp_idx < plane.components.size(); comp_idx++) {
                FinitePlane sub_plane;
                sub_plane.id = planes.size();  
                sub_plane.normal = plane.normal;
                sub_plane.d = plane.d;
                sub_plane.cloud = plane.components[comp_idx];
                sub_plane.point_count = sub_plane.cloud->points.size();
                sub_plane.center = plane.component_centers[comp_idx];
                sub_plane.min_x = plane.component_bbox_min[comp_idx].x();
                sub_plane.max_x = plane.component_bbox_max[comp_idx].x();
                sub_plane.min_y = plane.component_bbox_min[comp_idx].y();
                sub_plane.max_y = plane.component_bbox_max[comp_idx].y();
                sub_plane.min_z = plane.component_bbox_min[comp_idx].z();
                sub_plane.max_z = plane.component_bbox_max[comp_idx].z();

                        
                // 分配颜色
                int color_idx = sub_plane.id % colors.size();
                sub_plane.r = std::get<0>(colors[color_idx]);
                sub_plane.g = std::get<1>(colors[color_idx]);
                sub_plane.b = std::get<2>(colors[color_idx]);
                
                planes.push_back(sub_plane);
                
                std::cout << "    子平面 " << sub_plane.id << ": 中心=(" 
                          << sub_plane.center.x() << "," << sub_plane.center.y() << "," << sub_plane.center.z() 
                          << "), 点数=" << sub_plane.point_count 
                          << ", 颜色=RGB(" << sub_plane.r << "," << sub_plane.g << "," << sub_plane.b << ")" << std::endl;
            }
        } else if (plane.components.size() == 1) {
            // 只有一个分量，直接使用
            plane.cloud = plane.components[0];
            plane.center = plane.component_centers[0];
            plane.min_x = plane.component_bbox_min[0].x();
            plane.max_x = plane.component_bbox_max[0].x();
            plane.min_y = plane.component_bbox_min[0].y();
            plane.max_y = plane.component_bbox_max[0].y();
            plane.min_z = plane.component_bbox_min[0].z();
            plane.max_z = plane.component_bbox_max[0].z();
            
            // 分配颜色
            int color_idx = plane.id % colors.size();
            plane.r = std::get<0>(colors[color_idx]);
            plane.g = std::get<1>(colors[color_idx]);
            plane.b = std::get<2>(colors[color_idx]);
            
            planes.push_back(plane);
            
            std::cout << "[Plane " << plane.id << "] 单一分量: 中心=(" 
                      << plane.center.x() << "," << plane.center.y() << "," << plane.center.z() 
                      << "), 点数=" << plane.point_count 
                      << ", 颜色=RGB(" << plane.r << "," << plane.g << "," << plane.b << ")" << std::endl;
        } else {
            // 没有有效分量，跳过
            std::cout << "[Plane " << plane.id << "] 无有效分量，跳过" << std::endl;
        }
    }

    // ✅ 验证 ID 与向量索引一致
    for (size_t i = 0; i < planes.size(); i++) {
        if (planes[i].id != (int)i) {
            std::cout << "[修正] 平面索引: 向量索引=" << i 
                      << ", 原ID=" << planes[i].id << " -> 修正为 " << i << std::endl;
            planes[i].id = i;
        }
    }
    
    std::cout << "[PlaneExtractor] 共提取 " << planes.size() << " 个有限平面（含连通分量）" << std::endl;
    return planes;
}

bool PlaneExtractor::computeTriplePlaneIntersection(const FinitePlane& p1,
                                                     const FinitePlane& p2,
                                                     const FinitePlane& p3,
                                                     Eigen::Vector3f& intersection) {
    Eigen::Matrix3f A;
    A.row(0) = p1.normal;
    A.row(1) = p2.normal;
    A.row(2) = p3.normal;
    
    Eigen::Vector3f b(-p1.d, -p2.d, -p3.d);
    
    // 检查矩阵是否奇异（平面不独立）
    if (std::abs(A.determinant()) < 1e-6f) {
        return false;
    }
    
    intersection = A.inverse() * b;
    
    // 检查交点是否在平面边界内（放宽容差）
    float margin = 0.02f;
    if (!isPointInPlaneBounds(p1, intersection, margin) ||
        !isPointInPlaneBounds(p2, intersection, margin) ||
        !isPointInPlaneBounds(p3, intersection, margin)) {
        return false;
    }
    
    return true;
}

bool PlaneExtractor::computePlaneIntersectionLine(const FinitePlane& p1,
                                                   const FinitePlane& p2,
                                                   Eigen::Vector3f& direction,
                                                   Eigen::Vector3f& point_on_line) {
    direction = p1.normal.cross(p2.normal);
    if (direction.norm() < 1e-6) {
        return false;
    }
    direction.normalize();
    
    // 求解交线上一点
    Eigen::Matrix2f A;
    A << p1.normal.x(), p1.normal.y(),
         p2.normal.x(), p2.normal.y();
    
    Eigen::Vector2f b(-p1.d - p1.normal.z() * 0, -p2.d - p2.normal.z() * 0);
    
    if (std::abs(A.determinant()) > 1e-6) {
        Eigen::Vector2f xy = A.inverse() * b;
        point_on_line = Eigen::Vector3f(xy.x(), xy.y(), 0);
        return true;
    }
    
    // 如果不行，尝试其他组合
    Eigen::Matrix2f A_xz;
    A_xz << p1.normal.x(), p1.normal.z(),
            p2.normal.x(), p2.normal.z();
    Eigen::Vector2f b_xz(-p1.d - p1.normal.y() * 0, -p2.d - p2.normal.y() * 0);
    
    if (std::abs(A_xz.determinant()) > 1e-6) {
        Eigen::Vector2f xz = A_xz.inverse() * b_xz;
        point_on_line = Eigen::Vector3f(xz.x(), 0, xz.y());
        return true;
    }
    
    Eigen::Matrix2f A_yz;
    A_yz << p1.normal.y(), p1.normal.z(),
            p2.normal.y(), p2.normal.z();
    Eigen::Vector2f b_yz(-p1.d - p1.normal.x() * 0, -p2.d - p2.normal.x() * 0);
    
    if (std::abs(A_yz.determinant()) > 1e-6) {
        Eigen::Vector2f yz = A_yz.inverse() * b_yz;
        point_on_line = Eigen::Vector3f(0, yz.x(), yz.y());
        return true;
    }
    
    return false;
}

CloudPtr PlaneExtractor::sampleIntersectionLine(const FinitePlane& p1,
                                                 const FinitePlane& p2,
                                                 float step) {
    CloudPtr line_points(new Cloud);
    
    Eigen::Vector3f direction, point_on_line;
    if (!computePlaneIntersectionLine(p1, p2, direction, point_on_line)) {
        std::cout << "  [错误] 无法计算交线" << std::endl;
        return line_points;
    }
    
    // ✅ 扩大采样范围：使用两个平面边界的并集，再扩大一些
    float min_x = std::min(p1.min_x, p2.min_x);
    float max_x = std::max(p1.max_x, p2.max_x);
    float min_y = std::min(p1.min_y, p2.min_y);
    float max_y = std::max(p1.max_y, p2.max_y);
    float min_z = std::min(p1.min_z, p2.min_z);
    float max_z = std::max(p1.max_z, p2.max_z);
    
    // 扩大范围
    float range_x = (max_x - min_x) * 1.02f;
    float range_y = (max_y - min_y) * 1.02f;
    float range_z = (max_z - min_z) * 1.02f;
    float max_range = std::max({range_x, range_y, range_z});
    
    std::cout << "  [交线采样] 范围: " << -max_range << " ~ " << max_range 
              << ", 步长: " << step << std::endl;
    
    // 放宽容差
    float margin = 0.03f;  
    
    int sampled_points = 0;
    for (float t = -max_range; t <= max_range; t += step) {
        Eigen::Vector3f point = point_on_line + t * direction;
        
        // 检查点是否在两个平面的边界内（使用放宽容差）
        bool in_p1 = (point.x() >= p1.min_x - margin && point.x() <= p1.max_x + margin &&
                      point.y() >= p1.min_y - margin && point.y() <= p1.max_y + margin &&
                      point.z() >= p1.min_z - margin && point.z() <= p1.max_z + margin);
        
        bool in_p2 = (point.x() >= p2.min_x - margin && point.x() <= p2.max_x + margin &&
                      point.y() >= p2.min_y - margin && point.y() <= p2.max_y + margin &&
                      point.z() >= p2.min_z - margin && point.z() <= p2.max_z + margin);
        
        if (in_p1 && in_p2) {
            pcl::PointXYZ p;
            p.x = point.x(); p.y = point.y(); p.z = point.z();
            line_points->push_back(p);
            sampled_points++;
        }
    }
    
    std::cout << "  [交线采样] 获得 " << sampled_points << " 个点" << std::endl;
    
    // ✅ 如果还是没找到点，尝试只用一个平面的边界
    if (line_points->points.empty()) {
        std::cout << "  [交线采样] 放宽条件，只要求在一个平面内..." << std::endl;
        
        for (float t = -max_range; t <= max_range; t += step) {
            Eigen::Vector3f point = point_on_line + t * direction;
            
            bool in_p1 = (point.x() >= p1.min_x - margin && point.x() <= p1.max_x + margin &&
                          point.y() >= p1.min_y - margin && point.y() <= p1.max_y + margin &&
                          point.z() >= p1.min_z - margin && point.z() <= p1.max_z + margin);
            
            if (in_p1) {
                pcl::PointXYZ p;
                p.x = point.x(); p.y = point.y(); p.z = point.z();
                line_points->push_back(p);
            }
        }
        std::cout << "  [交线采样] 放宽后获得 " << line_points->points.size() << " 个点" << std::endl;
    }
    
    return line_points;
}

bool PlaneExtractor::isPointInPlaneBounds(const FinitePlane& plane, 
                                           const Eigen::Vector3f& point,
                                           float margin) {
    return (point.x() >= plane.min_x - margin && point.x() <= plane.max_x + margin &&
            point.y() >= plane.min_y - margin && point.y() <= plane.max_y + margin &&
            point.z() >= plane.min_z - margin && point.z() <= plane.max_z + margin);
}


// ========== 统一法向量朝外 ==========
void PlaneExtractor::orientNormalsOutward(std::vector<FinitePlane>& planes,
                                          const std::vector<Eigen::Vector3f>& camera_poses) {
    if (camera_poses.empty()) {
        std::cout << "[orientNormalsOutward] 无相机位姿，跳过法向量统一" << std::endl;
        return;
    }
    
    std::cout << "[orientNormalsOutward] 使用相机位姿统一法向量方向..." << std::endl;
    
    for (auto& plane : planes) {
        // 计算平均相机方向
        Eigen::Vector3f avg_cam_dir(0, 0, 0);
        int valid_cams = 0;
        
        for (const auto& cam : camera_poses) {
            Eigen::Vector3f to_cam = cam - plane.center;
            float dist = to_cam.norm();
            if (dist > 1e-6) {
                avg_cam_dir += to_cam / dist;
                valid_cams++;
            }
        }
        
        if (valid_cams == 0) {
            std::cout << "[Plane " << plane.id << "] 无有效相机，保持原方向" << std::endl;
            continue;
        }
        
        avg_cam_dir.normalize();
        
        // 计算法向量与相机方向的夹角
        float dot = plane.normal.dot(avg_cam_dir);
        
        if (dot < 0) {
            plane.normal = -plane.normal;
            plane.d = -plane.d;
            std::cout << "[Plane " << plane.id << "] 法向量已反转 (dot=" << dot << ")" << std::endl;
        } else {
            std::cout << "[Plane " << plane.id << "] 法向量方向正确 (dot=" << dot << ")" << std::endl;
        }
    }
    
    std::cout << "[orientNormalsOutward] 法向量统一完成" << std::endl;
}

