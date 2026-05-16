// plane_extractor.cpp
#include "plane_extractor.hpp"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
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
    std::vector<FinitePlane> raw_planes;
    CloudPtr remaining(new Cloud);
    *remaining = *cloud;
    
    std::cout << "[PlaneExtractor] 步骤1: 粗分割 (RANSAC)..." << std::endl;
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.005f);
    seg.setMaxIterations(2000);
    seg.setProbability(0.99);
    
    int max_planes = 20;
    int total_points = cloud->points.size();
    int classified_points = 0;
    
    for (int attempt = 0; attempt < max_planes; attempt++) {
        if (remaining->points.size() < (size_t)min_plane_points_) break;
        
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients coeff;
        
        seg.setInputCloud(remaining);
        seg.segment(*inliers, coeff);
        
        if (inliers->indices.size() < (size_t)min_plane_points_) continue;
        
        FinitePlane plane;
        plane.id = raw_planes.size();
        plane.normal = Eigen::Vector3f(coeff.values[0], coeff.values[1], coeff.values[2]);
        plane.normal.normalize();
        plane.d = coeff.values[3];
        
        // 提取平面点云
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        plane.cloud.reset(new Cloud);
        extract.setInputCloud(remaining);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane.cloud);
        
        // 从剩余点云中移除
        extract.setNegative(true);
        extract.filter(*remaining);
        
        plane.point_count = plane.cloud->points.size();
        classified_points += plane.point_count;
        
        computePlaneBounds(plane);
        raw_planes.push_back(plane);
        
        std::cout << "  平面 " << plane.id << ": 点数=" << plane.point_count 
                  << ", 累计覆盖率=" << (100.0f * classified_points / total_points) << "%" << std::endl;
    }
    
    float coverage = 100.0f * classified_points / total_points;
    std::cout << "[PlaneExtractor] 粗分割覆盖率: " << coverage << "%" << std::endl;
    
    // 步骤2: 3σ距离补全
    if (coverage < 99.9f) {
        std::cout << "[PlaneExtractor] 步骤2: 3σ距离补全..." << std::endl;
        completePlanesWithSigma(raw_planes, cloud);
    }
    
    // 步骤3: 有限连通域分割
    std::cout << "[PlaneExtractor] 步骤3: 有限连通域分割..." << std::endl;
    std::vector<FinitePlane> processed_planes;
    for (auto& plane : raw_planes) {
        if (!plane.cloud || plane.cloud->points.empty()) continue;
        
        // 连通分量分割
        segmentConnectedComponents(plane);
        
        if (plane.components.size() > 1) {
            for (size_t comp_idx = 0; comp_idx < plane.components.size(); comp_idx++) {
                FinitePlane sub_plane = plane;
                sub_plane.id = processed_planes.size();
                sub_plane.cloud = plane.components[comp_idx];
                sub_plane.center = plane.component_centers[comp_idx];
                sub_plane.min_x = plane.component_bbox_min[comp_idx].x();
                sub_plane.max_x = plane.component_bbox_max[comp_idx].x();
                sub_plane.min_y = plane.component_bbox_min[comp_idx].y();
                sub_plane.max_y = plane.component_bbox_max[comp_idx].y();
                sub_plane.min_z = plane.component_bbox_min[comp_idx].z();
                sub_plane.max_z = plane.component_bbox_max[comp_idx].z();
                sub_plane.point_count = sub_plane.cloud->points.size();
                
                computePlaneLocalFrame(sub_plane);
                computePlaneConvexHull(sub_plane);
                processed_planes.push_back(sub_plane);
            }
        } else if (plane.components.size() == 1) {
            plane.id = processed_planes.size();
            plane.cloud = plane.components[0];
            plane.center = plane.component_centers[0];
            plane.min_x = plane.component_bbox_min[0].x();
            plane.max_x = plane.component_bbox_max[0].x();
            plane.min_y = plane.component_bbox_min[0].y();
            plane.max_y = plane.component_bbox_max[0].y();
            plane.min_z = plane.component_bbox_min[0].z();
            plane.max_z = plane.component_bbox_max[0].z();
            
            computePlaneLocalFrame(plane);
            computePlaneConvexHull(plane);
            processed_planes.push_back(plane);
        }
    }
    
    std::cout << "[PlaneExtractor] 最终平面数: " << processed_planes.size() << std::endl;
    return processed_planes;
}

void PlaneExtractor::completePlanesWithSigma(std::vector<FinitePlane>& planes, CloudPtr cloud) {
    // 为每个平面计算距离的均值和标准差
    for (auto& plane : planes) {
        if (!plane.cloud || plane.cloud->points.empty()) continue;
        
        std::vector<float> distances;
        for (const auto& p : plane.cloud->points) {
            Eigen::Vector3f pt(p.x, p.y, p.z);
            float dist = std::abs(plane.normal.dot(pt) + plane.d);
            distances.push_back(dist);
        }
        
        // 计算均值和标准差
        float mean = 0, stddev = 0;
        for (float d : distances) mean += d;
        mean /= distances.size();
        for (float d : distances) stddev += (d - mean) * (d - mean);
        stddev = std::sqrt(stddev / distances.size());
        
        plane.distance_mean = mean;
        plane.distance_stddev = stddev;
        plane.sigma_threshold = mean + sigma_threshold_ * stddev;
        
        std::cout << "  平面 " << plane.id << ": 距离均值=" << mean 
                  << ", 标准差=" << stddev << ", 3σ阈值=" << plane.sigma_threshold << std::endl;
    }
    
    // 收集所有已分类点的索引（简化版：使用点云直接判断）
    // 注意：这里需要根据实际情况实现点分类标记
    
    int new_classified = 0;
    
    // 遍历原始点云中的每个点
    for (const auto& p : cloud->points) {
        Eigen::Vector3f pt(p.x, p.y, p.z);
        
        int best_plane = -1;
        float best_dist = FLT_MAX;
        
        // 找到最近的平面（在3σ阈值内）
        for (size_t j = 0; j < planes.size(); j++) {
            const auto& plane = planes[j];
            float dist = std::abs(plane.normal.dot(pt) + plane.d);
            
            if (dist < plane.sigma_threshold && dist < best_dist) {
                best_dist = dist;
                best_plane = j;
            }
        }
        
        if (best_plane >= 0) {
            // 检查点是否已经在平面中（简化：检查距离）
            bool already_in_plane = false;
            for (const auto& existing_pt : planes[best_plane].cloud->points) {
                float dx = existing_pt.x - p.x;
                float dy = existing_pt.y - p.y;
                float dz = existing_pt.z - p.z;
                if (std::sqrt(dx*dx + dy*dy + dz*dz) < 0.001f) {
                    already_in_plane = true;
                    break;
                }
            }
            
            if (!already_in_plane) {
                planes[best_plane].cloud->push_back(p);
                new_classified++;
            }
        }
    }
    
    std::cout << "[3σ补全] 新增分类点数: " << new_classified << std::endl;
    
    // 重新计算每个平面的边界
    for (auto& plane : planes) {
        computePlaneBounds(plane);
    }
}

std::vector<FinitePlane> PlaneExtractor::extractPlanesRegionGrowing(CloudPtr cloud) {
    std::vector<FinitePlane> raw_planes;
    
    std::cout << "[区域生长] 开始提取平面..." << std::endl;
    
    // 降采样
    CloudPtr downsampled = downsample(cloud);
    std::cout << "[区域生长] 降采样后点数: " << downsampled->points.size() << std::endl;
    
    // 去噪
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(downsampled);
    sor.setMeanK(20);
    sor.setStddevMulThresh(1.0);
    sor.filter(*downsampled);
    
    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(downsampled);
    ne.setKSearch(30);
    ne.compute(*normals);
    
    // 区域生长分割
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(min_plane_points_);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
    reg.setNumberOfNeighbours(30);
    reg.setSmoothModeFlag(true);
    reg.setCurvatureThreshold(curvature_threshold_);
    reg.setSmoothnessThreshold(normal_smoothness_threshold_ / 180.0f * M_PI);
    reg.setInputCloud(downsampled);
    reg.setInputNormals(normals);
    
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    
    std::cout << "[区域生长] 找到 " << clusters.size() << " 个平面簇" << std::endl;
    
    for (size_t i = 0; i < clusters.size(); i++) {
        if (clusters[i].indices.size() < (size_t)min_plane_points_) continue;
        
        FinitePlane raw_plane;
        raw_plane.id = raw_planes.size();  // 临时ID，后处理会重新分配
        raw_plane.cloud.reset(new Cloud);
        
        // 提取点云
        for (int idx : clusters[i].indices) {
            raw_plane.cloud->push_back(downsampled->points[idx]);
        }
        raw_plane.point_count = raw_plane.cloud->points.size();
        
        // ✅ 使用 RANSAC 拟合平面获取法向量和d（更稳定）
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients coeff;
        
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01f);
        seg.setInputCloud(raw_plane.cloud);
        seg.segment(*inliers, coeff);
        
        if (inliers->indices.size() > raw_plane.cloud->points.size() * 0.5) {
            raw_plane.normal = Eigen::Vector3f(coeff.values[0], coeff.values[1], coeff.values[2]);
            raw_plane.normal.normalize();
            raw_plane.d = coeff.values[3];
        } else {
            // 如果 RANSAC 失败，使用最小二乘法拟合
            Eigen::MatrixXf A(raw_plane.point_count, 3);
            Eigen::VectorXf b(raw_plane.point_count);
            for (size_t j = 0; j < raw_plane.cloud->points.size(); j++) {
                const auto& p = raw_plane.cloud->points[j];
                A(j, 0) = p.x;
                A(j, 1) = p.y;
                A(j, 2) = p.z;
                b(j) = 1.0f;
            }
            Eigen::Vector3f n = A.colPivHouseholderQr().solve(b);
            raw_plane.normal = n.normalized();
            raw_plane.d = -1.0f / n.norm();
        }
        
        // 计算边界（粗略，后处理会重新计算精确边界）
        computePlaneBounds(raw_plane);
        
        raw_planes.push_back(raw_plane);
        
        std::cout << "[区域生长] 提取原始平面 " << raw_plane.id << ": 点数=" << raw_plane.point_count << std::endl;
    }
    
    std::cout << "[区域生长] 共提取 " << raw_planes.size() << " 个原始平面" << std::endl;
    
    // ✅ 使用统一的后处理（连通分量分割 + 凸包计算）
    std::cout << "[区域生长] 开始后处理..." << std::endl;
    auto processed_planes = postProcessPlanes(raw_planes);
    
    // ✅ 合并相似平面
    std::cout << "[区域生长] 开始合并相似平面..." << std::endl;
    auto merged_planes = mergeSimilarPlanes(processed_planes);
    std::cout << "[区域生长] 合并后剩余 " << merged_planes.size() << " 个平面" << std::endl;
    
    return merged_planes;
}


// ========== 合并相似平面 ==========
std::vector<FinitePlane> PlaneExtractor::mergeSimilarPlanes(const std::vector<FinitePlane>& planes) {
    std::vector<FinitePlane> merged;
    std::vector<bool> used(planes.size(), false);
    
    // 法向量夹角阈值（度）
    float angle_threshold = 15.0f;   // 15度以内的认为平行
    float distance_threshold = 0.02f; // 2cm 以内的认为共面
    
    for (size_t i = 0; i < planes.size(); i++) {
        if (used[i]) continue;
        
        // 开始一个新组
        std::vector<int> group = { (int)i };
        used[i] = true;
        
        for (size_t j = i + 1; j < planes.size(); j++) {
            if (used[j]) continue;
            
            // 计算法向量夹角
            float dot = std::abs(planes[i].normal.dot(planes[j].normal));
            float angle = std::acos(std::min(1.0f, dot)) * 180.0f / M_PI;
            
            if (angle > angle_threshold) continue;
            
            // 计算平面距离
            float dist = std::abs(planes[i].normal.dot(planes[j].center) + planes[i].d);
            
            if (dist > distance_threshold) continue;
            
            // 相似平面，加入组
            group.push_back(j);
            used[j] = true;
        }
        
        if (group.size() == 1) {
            // 没有相似平面，直接加入
            merged.push_back(planes[i]);
        } else {
            // 合并组内所有平面
            FinitePlane merged_plane;
            merged_plane.id = merged.size();
            merged_plane.normal = planes[i].normal;
            merged_plane.d = planes[i].d;
            
            // 合并点云
            merged_plane.cloud.reset(new Cloud);
            for (int idx : group) {
                *merged_plane.cloud += *planes[idx].cloud;
            }
            merged_plane.point_count = merged_plane.cloud->points.size();
            
            // 重新计算边界
            computePlaneBounds(merged_plane);
            computePlaneLocalFrame(merged_plane);
            computePlaneConvexHull(merged_plane);
            
            // 重新拟合平面（使用合并后的点云）
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients coeff;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01f);
            seg.setInputCloud(merged_plane.cloud);
            seg.segment(*inliers, coeff);
            
            if (inliers->indices.size() > 0) {
                merged_plane.normal = Eigen::Vector3f(coeff.values[0], coeff.values[1], coeff.values[2]);
                merged_plane.normal.normalize();
                merged_plane.d = coeff.values[3];
            }
            
            // 分配颜色
            int color_idx = merged_plane.id % colors_.size();
            merged_plane.r = std::get<0>(colors_[color_idx]);
            merged_plane.g = std::get<1>(colors_[color_idx]);
            merged_plane.b = std::get<2>(colors_[color_idx]);
            
            merged.push_back(merged_plane);
            
            std::cout << "  [合并] " << group.size() << " 个平面合并为 1 个平面，总点数=" 
                      << merged_plane.point_count << std::endl;
        }
    }
    
    return merged;
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
    float margin = 0.03f;
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
    
    // ========== 1. 确定交线的实际范围（基于两个平面的交集） ==========
    // 方法：将两个平面的边界投影到交线方向上，取交集
    
    // 收集两个平面边界在交线方向上的投影范围
    auto getPlaneProjectionRange = [&](const FinitePlane& plane) -> std::pair<float, float> {
        float t_min = FLT_MAX, t_max = -FLT_MAX;
        
        // 采样平面边界上的关键点
        std::vector<Eigen::Vector3f> boundary_points;
        
        // 添加平面中心
        boundary_points.push_back(plane.center);
        
        // 添加平面角点（8个角点）
        boundary_points.push_back(Eigen::Vector3f(plane.min_x, plane.min_y, plane.min_z));
        boundary_points.push_back(Eigen::Vector3f(plane.min_x, plane.min_y, plane.max_z));
        boundary_points.push_back(Eigen::Vector3f(plane.min_x, plane.max_y, plane.min_z));
        boundary_points.push_back(Eigen::Vector3f(plane.min_x, plane.max_y, plane.max_z));
        boundary_points.push_back(Eigen::Vector3f(plane.max_x, plane.min_y, plane.min_z));
        boundary_points.push_back(Eigen::Vector3f(plane.max_x, plane.min_y, plane.max_z));
        boundary_points.push_back(Eigen::Vector3f(plane.max_x, plane.max_y, plane.min_z));
        boundary_points.push_back(Eigen::Vector3f(plane.max_x, plane.max_y, plane.max_z));
        
        // 如果有凸包，也添加凸包顶点
        for (const auto& hull_pt : plane.hull_2d) {
            Eigen::Vector3f pt = plane.origin + plane.local_x * hull_pt.x() + plane.local_y * hull_pt.y();
            boundary_points.push_back(pt);
        }
        
        // 计算投影
        for (const auto& pt : boundary_points) {
            float t = (pt - point_on_line).dot(direction);
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
        }
        
        return {t_min, t_max};
    };
    
    auto [t1_min, t1_max] = getPlaneProjectionRange(p1);
    auto [t2_min, t2_max] = getPlaneProjectionRange(p2);
    
    // 取交集作为实际交线范围
    float t_start = std::max(t1_min, t2_min);
    float t_end = std::min(t1_max, t2_max);
    
    if (t_start >= t_end) {
        std::cout << "  [交线采样] 两平面无重叠区域" << std::endl;
        return line_points;
    }
    
    // ========== 2. 自适应采样 ==========
    float length = t_end - t_start;
    int target_points = std::max(10, (int)(length / step));
    float adaptive_step = length / target_points;
    
    std::cout << "  [交线采样] 范围: [" << t_start << ", " << t_end 
              << "], 长度: " << length << "m, 目标点数: " << target_points << std::endl;
    
    // ========== 3. 采样并验证点是否在平面内 ==========
    float margin = 0.03f;
    std::vector<Eigen::Vector3f> sampled_points;
    
    for (int i = 0; i <= target_points; i++) {
        float t = t_start + i * adaptive_step;
        Eigen::Vector3f point = point_on_line + t * direction;
        
        // 使用精确边界判断（如果有凸包）
        bool in_p1 = isPointInPlaneBoundsExact(p1, point, margin);
        bool in_p2 = isPointInPlaneBoundsExact(p2, point, margin);
        
        if (in_p1 && in_p2) {
            pcl::PointXYZ p;
            p.x = point.x(); p.y = point.y(); p.z = point.z();
            line_points->push_back(p);
            sampled_points.push_back(point);
        }
    }
    
    // ========== 4. 如果采样点太少，尝试插值补充 ==========
    if (line_points->points.size() < 5 && sampled_points.size() >= 2) {
        std::cout << "  [交线采样] 采样点较少，尝试插值..." << std::endl;
        
        // 找到连续的有效区间
        std::vector<std::pair<float, float>> valid_ranges;
        float range_start = -1;
        bool in_range = false;
        
        for (int i = 0; i <= target_points; i++) {
            float t = t_start + i * adaptive_step;
            bool valid = false;
            for (const auto& p : sampled_points) {
                if (std::abs((p - (point_on_line + t * direction)).norm()) < 0.01f) {
                    valid = true;
                    break;
                }
            }
            
            if (valid && !in_range) {
                range_start = t;
                in_range = true;
            } else if (!valid && in_range) {
                valid_ranges.push_back({range_start, t - adaptive_step});
                in_range = false;
            }
        }
        if (in_range) {
            valid_ranges.push_back({range_start, t_end});
        }
        
        // 在有效区间内进行更密集的采样
        float fine_step = adaptive_step / 4;
        for (const auto& [r_start, r_end] : valid_ranges) {
            for (float t = r_start; t <= r_end; t += fine_step) {
                Eigen::Vector3f point = point_on_line + t * direction;
                
                bool in_p1 = isPointInPlaneBoundsExact(p1, point, margin * 2);
                bool in_p2 = isPointInPlaneBoundsExact(p2, point, margin * 2);
                
                if (in_p1 && in_p2) {
                    // 检查是否与已有点太近
                    bool too_close = false;
                    for (const auto& existing : line_points->points) {
                        if (std::abs(existing.x - point.x()) < fine_step &&
                            std::abs(existing.y - point.y()) < fine_step &&
                            std::abs(existing.z - point.z()) < fine_step) {
                            too_close = true;
                            break;
                        }
                    }
                    if (!too_close) {
                        pcl::PointXYZ p;
                        p.x = point.x(); p.y = point.y(); p.z = point.z();
                        line_points->push_back(p);
                    }
                }
            }
        }
    }
    
    // ========== 5. 去重和排序 ==========
    if (line_points->points.size() > 1) {
        // 按 t 值排序
        std::sort(line_points->points.begin(), line_points->points.end(),
            [&](const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
                float ta = (Eigen::Vector3f(a.x, a.y, a.z) - point_on_line).dot(direction);
                float tb = (Eigen::Vector3f(b.x, b.y, b.z) - point_on_line).dot(direction);
                return ta < tb;
            });
        
        // 去重
        auto last = std::unique(line_points->points.begin(), line_points->points.end(),
            [](const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
                return std::abs(a.x - b.x) < 1e-4f &&
                       std::abs(a.y - b.y) < 1e-4f &&
                       std::abs(a.z - b.z) < 1e-4f;
            });
        line_points->points.erase(last, line_points->points.end());
    }
    
    std::cout << "  [交线采样] 最终获得 " << line_points->points.size() << " 个点" << std::endl;
    
    return line_points;
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

// ========== 精确判断点是否在平面边界内（使用2D凸包）==========
bool PlaneExtractor::isPointInPlaneBoundsExact(const FinitePlane& plane, 
                                                const Eigen::Vector3f& point, 
                                                float margin) {
    // 1. 快速AABB筛选（先过滤掉明显不在内的点）
    if (point.x() < plane.min_x - margin || point.x() > plane.max_x + margin ||
        point.y() < plane.min_y - margin || point.y() > plane.max_y + margin ||
        point.z() < plane.min_z - margin || point.z() > plane.max_z + margin) {
        return false;
    }
    
    // 2. 如果没有凸包（点云太小或计算失败），使用AABB结果
    if (plane.hull_2d.empty()) {
        return true;
    }
    
    // 3. 将点投影到平面局部坐标系
    Eigen::Vector3f local = point - plane.origin;
    float u = local.dot(plane.local_x);
    float v = local.dot(plane.local_y);
    
    // 4. 射线法判断点是否在凸包内
    bool inside = false;
    size_t n = plane.hull_2d.size();
    
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const auto& pi = plane.hull_2d[i];
        const auto& pj = plane.hull_2d[j];
        
        // 检查射线是否与边相交
        if (((pi.y() > v) != (pj.y() > v)) &&
            (u < (pj.x() - pi.x()) * (v - pi.y()) / (pj.y() - pi.y()) + pi.x())) {
            inside = !inside;
        }
    }
    
    // 5. 考虑容差：如果点在凸包外但距离很近，也认为在内
    if (!inside) {
        // 计算点到凸包的最短距离（简化：检查到每条边的距离）
        float min_dist = FLT_MAX;
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const auto& pi = plane.hull_2d[i];
            const auto& pj = plane.hull_2d[j];
            
            Eigen::Vector2f edge = pj - pi;
            Eigen::Vector2f to_point = Eigen::Vector2f(u, v) - pi;
            
            float edge_len = edge.norm();
            if (edge_len < 1e-6f) continue;
            
            float t = to_point.dot(edge) / (edge_len * edge_len);
            t = std::max(0.0f, std::min(1.0f, t));
            
            Eigen::Vector2f closest = pi + t * edge;
            float dist = (Eigen::Vector2f(u, v) - closest).norm();
            min_dist = std::min(min_dist, dist);
        }
        
        // 如果距离在容差内，也认为在边界内
        if (min_dist < margin) {
            inside = true;
        }
    }
    
    return inside;
}

// ========== 简单边界判断（使用AABB）==========
bool PlaneExtractor::isPointInPlaneBounds(const FinitePlane& plane, 
                                           const Eigen::Vector3f& point,
                                           float margin) {
    return (point.x() >= plane.min_x - margin && point.x() <= plane.max_x + margin &&
            point.y() >= plane.min_y - margin && point.y() <= plane.max_y + margin &&
            point.z() >= plane.min_z - margin && point.z() <= plane.max_z + margin);
}

// ========== 计算平面的局部坐标系 ==========
void PlaneExtractor::computePlaneLocalFrame(FinitePlane& plane) {
    if (!plane.cloud || plane.cloud->points.empty()) return;
    
    // 法向量为Z轴
    Eigen::Vector3f n = plane.normal.normalized();
    
    // 选择一个参考轴（不与法向量平行即可）
    Eigen::Vector3f ref(1, 0, 0);
    if (std::abs(n.dot(ref)) > 0.9999f) {
        ref = Eigen::Vector3f(0, 1, 0);
    }
    
    // 计算局部X轴 = ref - (ref·n)n
    plane.local_x = ref - n * n.dot(ref);
    if (plane.local_x.norm() > 1e-6f) {
        plane.local_x.normalize();
    } else {
        plane.local_x = Eigen::Vector3f(1, 0, 0);
    }
    
    // 局部Y轴 = n × X
    plane.local_y = n.cross(plane.local_x);
    plane.local_y.normalize();
    
    // 原点取平面中心
    plane.origin = plane.center;
    
    std::cout << "    [局部坐标系] 原点=(" << plane.origin.x() << "," << plane.origin.y() << "," << plane.origin.z() 
              << "), X轴=(" << plane.local_x.x() << "," << plane.local_x.y() << "," << plane.local_x.z() 
              << "), Y轴=(" << plane.local_y.x() << "," << plane.local_y.y() << "," << plane.local_y.z() << ")" << std::endl;
}

// ========== 计算平面点云的2D凸包 ==========
void PlaneExtractor::computePlaneConvexHull(FinitePlane& plane) {
    if (!plane.cloud || plane.cloud->points.empty()) return;
    
    // 投影到2D
    std::vector<Eigen::Vector2f> points_2d;
    for (const auto& p : plane.cloud->points) {
        Eigen::Vector3f pt(p.x, p.y, p.z);
        Eigen::Vector3f local = pt - plane.origin;
        float u = local.dot(plane.local_x);
        float v = local.dot(plane.local_y);
        points_2d.push_back(Eigen::Vector2f(u, v));
    }
    
    if (points_2d.size() < 3) {
        plane.hull_2d.clear();
        return;
    }
    
    // 找最低最左的点
    int start_idx = 0;
    for (size_t i = 1; i < points_2d.size(); i++) {
        if (points_2d[i].y() < points_2d[start_idx].y() ||
            (points_2d[i].y() == points_2d[start_idx].y() && points_2d[i].x() < points_2d[start_idx].x())) {
            start_idx = i;
        }
    }
    
    std::swap(points_2d[0], points_2d[start_idx]);
    Eigen::Vector2f start = points_2d[0];
    
    // 按极角排序
    std::sort(points_2d.begin() + 1, points_2d.end(),
        [&start](const Eigen::Vector2f& a, const Eigen::Vector2f& b) {
            float cross = (a.x() - start.x()) * (b.y() - start.y()) - (a.y() - start.y()) * (b.x() - start.x());
            if (std::abs(cross) < 1e-6f) {
                return (a - start).squaredNorm() < (b - start).squaredNorm();
            }
            return cross > 0;
        });
    
    // Graham scan 计算凸包
    std::vector<Eigen::Vector2f> hull;
    for (const auto& p : points_2d) {
        while (hull.size() >= 2) {
            Eigen::Vector2f& a = hull[hull.size() - 2];
            Eigen::Vector2f& b = hull[hull.size() - 1];
            float cross = (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x());
            if (cross <= 0) {
                hull.pop_back();
            } else {
                break;
            }
        }
        hull.push_back(p);
    }
    
    plane.hull_2d = hull;
    
    std::cout << "    [凸包] 顶点数=" << hull.size() << std::endl;
}


// ========== 统一的平面后处理：连通分量分割 + 索引修正 ==========
std::vector<FinitePlane> PlaneExtractor::postProcessPlanes(const std::vector<FinitePlane>& input_planes) {
    std::vector<FinitePlane> result_planes;
    
    std::cout << "[后处理] 开始处理 " << input_planes.size() << " 个输入平面" << std::endl;
    
    // 预定义颜色列表
    std::vector<std::tuple<int, int, int>> colors = colors_;
    
    for (const auto& plane : input_planes) {
        if (!plane.cloud || plane.cloud->points.empty()) continue;
        
        // 创建临时平面对象用于连通分量分割
        FinitePlane temp_plane = plane;
        
        // 执行连通分量分割
        segmentConnectedComponents(temp_plane);
        
        if (temp_plane.components.size() > 1) {
            // 多个连通分量，分裂成多个独立平面
            std::cout << "  [分裂] 原始平面 " << plane.id << " 分裂为 " 
                      << temp_plane.components.size() << " 个独立平面" << std::endl;
            
            for (size_t comp_idx = 0; comp_idx < temp_plane.components.size(); comp_idx++) {
                FinitePlane sub_plane;
                sub_plane.id = result_planes.size();  // 使用 result_planes.size() 作为新ID
                sub_plane.normal = plane.normal;
                sub_plane.d = plane.d;
                sub_plane.cloud = temp_plane.components[comp_idx];
                sub_plane.point_count = sub_plane.cloud->points.size();
                sub_plane.center = temp_plane.component_centers[comp_idx];
                
                // 边界
                sub_plane.min_x = temp_plane.component_bbox_min[comp_idx].x();
                sub_plane.max_x = temp_plane.component_bbox_max[comp_idx].x();
                sub_plane.min_y = temp_plane.component_bbox_min[comp_idx].y();
                sub_plane.max_y = temp_plane.component_bbox_max[comp_idx].y();
                sub_plane.min_z = temp_plane.component_bbox_min[comp_idx].z();
                sub_plane.max_z = temp_plane.component_bbox_max[comp_idx].z();
                
                // 局部坐标系和凸包
                computePlaneLocalFrame(sub_plane);
                computePlaneConvexHull(sub_plane);
                
                // 分配颜色
                int color_idx = sub_plane.id % colors.size();
                sub_plane.r = std::get<0>(colors[color_idx]);
                sub_plane.g = std::get<1>(colors[color_idx]);
                sub_plane.b = std::get<2>(colors[color_idx]);
                
                result_planes.push_back(sub_plane);
                
                std::cout << "    子平面 " << sub_plane.id << ": 中心=(" 
                          << sub_plane.center.x() << "," << sub_plane.center.y() << "," << sub_plane.center.z() 
                          << "), 点数=" << sub_plane.point_count 
                          << ", 颜色=RGB(" << sub_plane.r << "," << sub_plane.g << "," << sub_plane.b << ")" << std::endl;
            }
        } else if (temp_plane.components.size() == 1) {
            // 单一分量，直接使用
            FinitePlane single_plane = plane;
            single_plane.id = result_planes.size();  // 使用 result_planes.size() 作为新ID
            single_plane.cloud = temp_plane.components[0];
            single_plane.center = temp_plane.component_centers[0];
            single_plane.min_x = temp_plane.component_bbox_min[0].x();
            single_plane.max_x = temp_plane.component_bbox_max[0].x();
            single_plane.min_y = temp_plane.component_bbox_min[0].y();
            single_plane.max_y = temp_plane.component_bbox_max[0].y();
            single_plane.min_z = temp_plane.component_bbox_min[0].z();
            single_plane.max_z = temp_plane.component_bbox_max[0].z();
            
            // 局部坐标系和凸包
            computePlaneLocalFrame(single_plane);
            computePlaneConvexHull(single_plane);
            
            // 分配颜色
            int color_idx = single_plane.id % colors.size();
            single_plane.r = std::get<0>(colors[color_idx]);
            single_plane.g = std::get<1>(colors[color_idx]);
            single_plane.b = std::get<2>(colors[color_idx]);
            
            result_planes.push_back(single_plane);
            
            std::cout << "[平面 " << single_plane.id << "] 单一分量: 中心=(" 
                      << single_plane.center.x() << "," << single_plane.center.y() << "," << single_plane.center.z() 
                      << "), 点数=" << single_plane.point_count 
                      << ", 颜色=RGB(" << single_plane.r << "," << single_plane.g << "," << single_plane.b << ")" << std::endl;
        }
        // 无有效分量的情况跳过
    }
    
    // ✅ 验证 ID 与向量索引一致
    for (size_t i = 0; i < result_planes.size(); i++) {
        if (result_planes[i].id != (int)i) {
            std::cout << "[修正] 平面索引: 向量索引=" << i 
                      << ", 原ID=" << result_planes[i].id << " -> 修正为 " << i << std::endl;
            result_planes[i].id = i;
        }
    }
    
    std::cout << "[后处理] 完成，共 " << result_planes.size() << " 个平面" << std::endl;
    
    return result_planes;
}
