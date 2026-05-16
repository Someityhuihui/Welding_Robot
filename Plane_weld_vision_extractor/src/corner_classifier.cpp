// corner_classifier.cpp
#include "corner_classifier.hpp"
#include "plane_extractor.hpp" 

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cfloat>      
#include <set>         
#include <algorithm>   
#include <iomanip>

// 在构造函数中初始化成员变量
CornerClassifier::CornerClassifier() : cloud_(new Cloud) {
    cloud_center_.setZero();
    cloud_bbox_min_.setZero();
    cloud_bbox_max_.setZero();
    shrunk_bbox_min_.setZero();
    shrunk_bbox_max_.setZero();
    filter_boundary_corners_ = true;
    boundary_margin_mm_ = 10.0f;
}

CornerClassifier::~CornerClassifier() {}

void CornerClassifier::setPointCloud(CloudPtr cloud) {
    cloud_ = cloud;
    
    // 计算点云重心和包围盒
    cloud_center_.setZero();
    cloud_bbox_min_ = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
    cloud_bbox_max_ = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const auto& p : cloud->points) {
        cloud_center_ += Eigen::Vector3f(p.x, p.y, p.z);
        cloud_bbox_min_ = cloud_bbox_min_.cwiseMin(Eigen::Vector3f(p.x, p.y, p.z));
        cloud_bbox_max_ = cloud_bbox_max_.cwiseMax(Eigen::Vector3f(p.x, p.y, p.z));
    }
    cloud_center_ /= cloud->points.size();
}

void CornerClassifier::setCameraPoses(const std::vector<Eigen::Vector3f>& poses) {
    camera_poses_ = poses;
}

Eigen::Vector3f CornerClassifier::vectorToPlaneCenter(const CornerPoint& corner, const FinitePlane& plane) {
    return plane.center - corner.position;
}

float CornerClassifier::computeAngle(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
    float dot = v1.dot(v2);
    dot = std::max(-1.0f, std::min(1.0f, dot));
    return std::acos(dot) * 180.0f / M_PI;
}

bool CornerClassifier::isDirectionOutward(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction) {
    if (!cloud_ || cloud_->points.empty()) return true;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_);
    
    float step = 0.005f;
    int max_steps = 50;
    int hit_count = 0;
    int sample_count = 0;
    
    for (int i = 1; i <= max_steps; i++) {
        Eigen::Vector3f sample = origin + direction * step * i;
        
        // 检查是否超出包围盒
        if (sample.x() < cloud_bbox_min_.x() - 0.05f || sample.x() > cloud_bbox_max_.x() + 0.05f ||
            sample.y() < cloud_bbox_min_.y() - 0.05f || sample.y() > cloud_bbox_max_.y() + 0.05f ||
            sample.z() < cloud_bbox_min_.z() - 0.05f || sample.z() > cloud_bbox_max_.z() + 0.05f) {
            break;
        }
        
        pcl::PointXYZ search_pt;
        search_pt.x = sample.x();
        search_pt.y = sample.y();
        search_pt.z = sample.z();
        
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree.radiusSearch(search_pt, 0.01f, indices, distances);
        
        sample_count++;
        if (!indices.empty()) {
            hit_count++;
        }
    }
    
    float hit_ratio = (sample_count > 0) ? (float)hit_count / sample_count : 1.0f;
    return hit_ratio < outward_threshold_;
}

bool CornerClassifier::classifyByVectorSum(const CornerPoint& corner, const std::vector<FinitePlane>& planes) {
    Eigen::Vector3f sum(0, 0, 0);
    
    for (int pid : corner.plane_ids) {
        const auto& plane = planes[pid];
        Eigen::Vector3f to_center = vectorToPlaneCenter(corner, plane);
        sum += to_center;
    }
    
    sum.normalize();
    return isDirectionOutward(corner.position, sum);
}

bool CornerClassifier::classifyByCameraPose(const CornerPoint& corner, const std::vector<FinitePlane>& planes) {
    if (camera_poses_.empty()) return false;
    
    // 计算三个平面中心的加权和
    Eigen::Vector3f weighted_sum(0, 0, 0);
    for (int pid : corner.plane_ids) {
        const auto& plane = planes[pid];
        Eigen::Vector3f to_center = vectorToPlaneCenter(corner, plane);
        float weight = (plane.max_x - plane.min_x) * (plane.max_y - plane.min_y);
        weighted_sum += to_center * weight;
    }
    weighted_sum.normalize();
    
    // 计算平均相机方向
    Eigen::Vector3f avg_camera_dir(0, 0, 0);
    for (const auto& cam_pose : camera_poses_) {
        Eigen::Vector3f to_camera = cam_pose - corner.position;
        if (to_camera.norm() > 1e-6) {
            to_camera.normalize();
            avg_camera_dir += to_camera;
        }
    }
    avg_camera_dir.normalize();
    
    float angle = computeAngle(weighted_sum, avg_camera_dir);
    
    if (angle <= 60.0f) return true;   // 内凹角
    if (angle >= 120.0f) return false; // 外凸角
    return classifyByVectorSum(corner, planes); // 不确定时用向量和法
}

bool CornerClassifier::classifyByNormalDistribution(const CornerPoint& corner, const std::vector<FinitePlane>& planes) {
    if (!cloud_ || cloud_->points.empty()) return false;
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_);
    
    pcl::PointXYZ search_pt;
    search_pt.x = corner.position.x();
    search_pt.y = corner.position.y();
    search_pt.z = corner.position.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    kdtree.radiusSearch(search_pt, 0.02f, indices, distances);
    
    if (indices.size() < 10) return false;
    
    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud_);
    ne.setKSearch(20);
    ne.compute(*normals);
    
    int outward_count = 0;
    for (int idx : indices) {
        Eigen::Vector3f normal(normals->points[idx].normal_x,
                               normals->points[idx].normal_y,
                               normals->points[idx].normal_z);
        
        Eigen::Vector3f to_point(cloud_->points[idx].x - cloud_center_.x(),
                                 cloud_->points[idx].y - cloud_center_.y(),
                                 cloud_->points[idx].z - cloud_center_.z());
        to_point.normalize();
        
        if (normal.dot(to_point) > 0) {
            outward_count++;
        }
    }
    
    float outward_ratio = (float)outward_count / indices.size();
    return (outward_ratio > 0.3f && outward_ratio < 0.7f);
}

// CornerClassifier::classifyCorners 函数(v2)，使用新的评分方法：相机可见性 + 向量夹角评分法，增强分类效果，并输出更详细的日志信息
// std::vector<CornerPoint> CornerClassifier::classifyCorners(
//     const std::vector<FinitePlane>& planes,
//     const std::vector<Eigen::Vector3f>& intersection_points,
//     const std::vector<std::tuple<int, int, int>>& intersection_plane_indices) {
    
//     std::vector<CornerPoint> corners;
    
//     std::cout << "\n========================================" << std::endl;
//     std::cout << "[CornerClassifier] 角点分类" << std::endl;
//     std::cout << "========================================" << std::endl;
//     std::cout << "总角点数: " << intersection_points.size() << std::endl;
//     std::cout << "总相机数: " << camera_poses_.size() << std::endl;
//     std::cout << "判定方法: 从平面点云随机采样，所有采样组合的局部坐标分量都 > 0.01" << std::endl;
//     std::cout << "内凹角判定: 有相机可见（≥1个相机满足条件）" << std::endl;
//     std::cout << "========================================\n" << std::endl;
    
//     int concave_count = 0;
//     int convex_count = 0;
    
//     for (size_t idx = 0; idx < intersection_points.size(); idx++) {
//         const auto& pt = intersection_points[idx];
//         auto [i, j, k] = intersection_plane_indices[idx];
        
//         // 验证索引有效性
//         if (i >= (int)planes.size() || j >= (int)planes.size() || k >= (int)planes.size()) {
//             std::cerr << "[错误] 索引超出范围: " << i << "," << j << "," << k << std::endl;
//             continue;
//         }
        
//         CornerPoint corner;
//         corner.position = pt;
//         corner.plane_ids = {i, j, k};
        
//         std::cout << "\n[角点 " << idx << "] " << std::endl;
//         std::cout << "  位置: (" << std::fixed << std::setprecision(4)
//                   << pt.x() << ", " << pt.y() << ", " << pt.z() << ")" << std::endl;
//         std::cout << "  平面: (" << i << "," << j << "," << k << ")" << std::endl;
        
//         // 统计可见相机数量
//         int visible_cameras = countCamerasSeeingConcaveCorner(corner, planes, camera_poses_);
        
//         // 判定内凹角：只要有至少1个相机可见，就是内凹角（焊缝）
//         // 因为真正的内凹角，相机从内部看，所有采样组合都会满足分量>0
//         if (visible_cameras > 0) {
//             corner.is_concave = true;
//             concave_count++;
//             std::cout << "  ✅ 判定结果: 内凹角 (焊缝) - 有 " << visible_cameras << " 个相机可见" << std::endl;
//         } else {
//             corner.is_concave = false;
//             convex_count++;
//             std::cout << "  ❌ 判定结果: 外凸角 (边角) - 无相机可见" << std::endl;
//         }
        
//         // 计算置信度（基于可见相机比例）
//         corner.confidence = (float)visible_cameras / camera_poses_.size();
        
//         // 计算内凹方向（用于焊缝生长）
//         if (corner.is_concave) {
//             // 内凹角：方向指向内部（三个平面中心的反方向）
//             Eigen::Vector3f sum_center(0, 0, 0);
//             for (int pid : corner.plane_ids) {
//                 sum_center += planes[pid].center;
//             }
//             corner.inward_direction = (corner.position - sum_center / 3).normalized();
//         } else {
//             // 外凸角：方向指向外部
//             Eigen::Vector3f sum_center(0, 0, 0);
//             for (int pid : corner.plane_ids) {
//                 sum_center += planes[pid].center;
//             }
//             corner.inward_direction = (sum_center / 3 - corner.position).normalized();
//         }
        
//         corners.push_back(corner);
//     }
    
//     std::cout << "\n========================================" << std::endl;
//     std::cout << "[角点统计] 内凹(焊缝)=" << concave_count 
//               << ", 外凸(边角)=" << convex_count << std::endl;
//     std::cout << "========================================\n" << std::endl;
    
//     // 保存角点信息
//     saveCornersToFile(corners, "corners_info.txt");
    
//     return corners;
// }


WeldSeam CornerClassifier::createWeldFromIntersection(const FinitePlane& p1, const FinitePlane& p2, int id) {
    WeldSeam seam;
    seam.id = id;
    seam.plane_id1 = p1.id;
    seam.plane_id2 = p2.id;
    seam.is_corner_weld = false;
    
    std::cout << "  [创建焊缝] 平面 " << p1.id << " 和 " << p2.id << std::endl;
    std::cout << "    平面1: 中心=(" << p1.center.x() << "," << p1.center.y() << "," << p1.center.z() 
              << "), 范围 X[" << p1.min_x << "," << p1.max_x << "]" << std::endl;
    std::cout << "    平面2: 中心=(" << p2.center.x() << "," << p2.center.y() << "," << p2.center.z() 
              << "), 范围 X[" << p2.min_x << "," << p2.max_x << "]" << std::endl;
    
    PlaneExtractor extractor;
    extractor.setDistanceThreshold(0.005f);
    extractor.setMinPlanePoints(100);
    
    CloudPtr line_points = extractor.sampleIntersectionLine(p1, p2, path_spacing_);
    
    if (line_points->points.empty()) {
        std::cout << "  [创建焊缝] 交线点云为空" << std::endl;
        return seam;
    }
    
    std::cout << "  [创建焊缝] 交线点云点数: " << line_points->points.size() << std::endl;
    
    // 采样路径点
    for (const auto& p : line_points->points) {
        seam.path.push_back(Eigen::Vector3f(p.x, p.y, p.z));
    }
    
    // 计算焊缝长度
    seam.length = 0;
    for (size_t i = 1; i < seam.path.size(); i++) {
        seam.length += (seam.path[i] - seam.path[i-1]).norm();
    }
    
    std::cout << "  [创建焊缝] 路径点数: " << seam.path.size() << ", 长度: " << seam.length * 1000 << " mm" << std::endl;
    
    // 采样姿态
    samplePathOnLine(line_points, seam);
    
    return seam;
}

void CornerClassifier::samplePathOnLine(CloudPtr line_points, WeldSeam& seam) {
    if (line_points->points.empty()) return;
    
    // 计算主方向
    pcl::PointXYZ min_pt, max_pt;
    min_pt.x = min_pt.y = min_pt.z = FLT_MAX;
    max_pt.x = max_pt.y = max_pt.z = -FLT_MAX;
    
    for (const auto& p : line_points->points) {
        if (p.x < min_pt.x) min_pt.x = p.x;
        if (p.x > max_pt.x) max_pt.x = p.x;
        if (p.y < min_pt.y) min_pt.y = p.y;
        if (p.y > max_pt.y) max_pt.y = p.y;
        if (p.z < min_pt.z) min_pt.z = p.z;
        if (p.z > max_pt.z) max_pt.z = p.z;
    }
    
    Eigen::Vector3f direction;
    float range_x = max_pt.x - min_pt.x;
    float range_y = max_pt.y - min_pt.y;
    float range_z = max_pt.z - min_pt.z;
    
    if (range_x >= range_y && range_x >= range_z) {
        direction = Eigen::Vector3f(1, 0, 0);
    } else if (range_y >= range_x && range_y >= range_z) {
        direction = Eigen::Vector3f(0, 1, 0);
    } else {
        direction = Eigen::Vector3f(0, 0, 1);
    }
    
    // 对路径点排序
    std::vector<std::pair<float, Eigen::Vector3f>> sorted;
    for (const auto& pos : seam.path) {
        float coord = direction.dot(pos);
        sorted.push_back({coord, pos});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // 重新排序路径点
    seam.path.clear();
    for (const auto& s : sorted) {
        seam.path.push_back(s.second);
    }
    
    // 计算法向量和姿态
    for (size_t i = 0; i < seam.path.size(); i++) {
        // 法向量暂时用垂直于焊接方向的方向
        Eigen::Vector3f perp;
        if (std::abs(direction.x()) < 0.9f) {
            perp = direction.cross(Eigen::Vector3f(1, 0, 0)).normalized();
        } else {
            perp = direction.cross(Eigen::Vector3f(0, 1, 0)).normalized();
        }
        seam.normals.push_back(perp);
        
        // 焊枪姿态（绕焊接方向的旋转）
        float angle_rad = weld_angle_ * M_PI / 180.0f;
        Eigen::AngleAxisf roll_angle(angle_rad, direction);
        Eigen::Quaternionf q(roll_angle);
        seam.orientations.push_back(q);
    }
}



bool CornerClassifier::exportWeldSeamsToCSV(const std::vector<WeldSeam>& seams, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[CornerClassifier] 无法保存焊缝: " << filename << std::endl;
        return false;
    }
    
    file << "weld_id,x,y,z,nx,ny,nz,ox,oy,oz,ow" << std::endl;
    
    for (const auto& seam : seams) {
        for (size_t i = 0; i < seam.path.size(); i++) {
            const auto& pos = seam.path[i];
            const auto& normal = seam.normals[i];
            const auto& q = seam.orientations[i];
            
            file << seam.id << ","
                 << pos.x() << "," << pos.y() << "," << pos.z() << ","
                 << normal.x() << "," << normal.y() << "," << normal.z() << ","
                 << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << std::endl;
        }
    }
    
    file.close();
    std::cout << "[CornerClassifier] 保存焊缝: " << filename << std::endl;
    return true;
}


// 添加新函数：检查点是否在点云边界内
bool CornerClassifier::isPointInCloudBounds(const Eigen::Vector3f& point, float margin) {
    if (!cloud_ || cloud_->points.empty()) return true;
    
    // 检查点是否在点云包围盒内
    return (point.x() >= cloud_bbox_min_.x() - margin && point.x() <= cloud_bbox_max_.x() + margin &&
            point.y() >= cloud_bbox_min_.y() - margin && point.y() <= cloud_bbox_max_.y() + margin &&
            point.z() >= cloud_bbox_min_.z() - margin && point.z() <= cloud_bbox_max_.z() + margin);
}

// 添加新函数：过滤焊缝超出点云范围的部分
void CornerClassifier::filterWeldSeamByBounds(WeldSeam& seam) {
    std::vector<Eigen::Vector3f> filtered_path;
    std::vector<Eigen::Vector3f> filtered_normals;
    std::vector<Eigen::Quaternionf> filtered_orientations;
    
    for (size_t i = 0; i < seam.path.size(); i++) {
        if (isPointInCloudBounds(seam.path[i])) {
            filtered_path.push_back(seam.path[i]);
            if (i < seam.normals.size()) filtered_normals.push_back(seam.normals[i]);
            if (i < seam.orientations.size()) filtered_orientations.push_back(seam.orientations[i]);
        }
    }
    
    // 更新焊缝
    if (filtered_path.size() >= 2) {
        seam.path = filtered_path;
        seam.normals = filtered_normals;
        seam.orientations = filtered_orientations;
        
        // 重新计算长度
        seam.length = 0;
        for (size_t i = 1; i < seam.path.size(); i++) {
            seam.length += (seam.path[i] - seam.path[i-1]).norm();
        }
    } else {
        // 如果过滤后点数不足，标记为空
        seam.path.clear();
    }
}

// extractWeldSeams 函数v2：根据提取模式从角点提取焊缝，长边焊缝，或者都提取
std::vector<WeldSeam> CornerClassifier::extractWeldSeams(const std::vector<FinitePlane>& planes,
                                                          const std::vector<CornerPoint>& corners) {
    std::vector<WeldSeam> seams;
    std::set<std::pair<int, int>> added_seams;
    
    std::cout << "\n[焊缝提取] 模式: ";
    switch(extraction_mode_) {
        case MODE_CORNER_ONLY:
            std::cout << "边角模式 (只提取三平面相交焊缝)" << std::endl;
            break;
        case MODE_LONG_ONLY:
            std::cout << "长条模式 (只提取两平面相交长焊缝)" << std::endl;
            break;
        case MODE_BOTH:
            std::cout << "混合模式 (提取所有焊缝)" << std::endl;
            break;
    }
    std::cout << "最小焊缝长度: " << min_weld_length_ * 1000 << " mm" << std::endl;
    
    // ========== 1. 边角模式：从内凹角提取三条焊缝 ==========
    if (extraction_mode_ == MODE_CORNER_ONLY || extraction_mode_ == MODE_BOTH) {
        std::cout << "\n[边角模式] 处理内凹角..." << std::endl;
        
        int concave_corners = 0;
        for (const auto& corner : corners) {
            if (!corner.is_concave) continue;
            
            concave_corners++;
            std::cout << "[内凹角 " << concave_corners << "] 平面: ";
            for (int pid : corner.plane_ids) std::cout << pid << " ";
            std::cout << std::endl;
            
            // 提取三条交线（每条边角对应三条焊缝）
            int weld_count_this_corner = 0;
            for (size_t i = 0; i < corner.plane_ids.size(); i++) {
                int pid1 = corner.plane_ids[i];
                int pid2 = corner.plane_ids[(i+1) % corner.plane_ids.size()];
                
                auto key = std::make_pair(std::min(pid1, pid2), std::max(pid1, pid2));
                if (added_seams.find(key) != added_seams.end()) {
                    std::cout << "  交线 " << pid1 << "-" << pid2 << " 已存在，跳过" << std::endl;
                    continue;
                }
                
                WeldSeam seam = createWeldFromIntersection(planes[pid1], planes[pid2], seams.size());
                
                // 过滤超出点云范围的点
                filterWeldSeamByBounds(seam);
                
                if (!seam.path.empty() && seam.length >= min_weld_length_) {
                    seam.is_corner_weld = true;
                    seams.push_back(seam);
                    added_seams.insert(key);
                    weld_count_this_corner++;
                    std::cout << "  ✅ 焊缝 " << seam.id << ": 平面 " << pid1 << " - " << pid2 
                              << ", 路径点 " << seam.path.size() << ", 长度 " << seam.length * 1000 << " mm" << std::endl;
                } else {
                    std::cout << "  ⚠️ 焊缝 " << pid1 << "-" << pid2 << " 无效 (点数=" 
                              << seam.path.size() << ", 长度=" << seam.length * 1000 << "mm)" << std::endl;
                }
            }
            
            // 边角模式：每个内凹角必须有三条焊缝，如果不足，尝试强制生成
            if (extraction_mode_ == MODE_CORNER_ONLY && weld_count_this_corner < 3) {
                std::cout << "  ⚠️ 该内凹角只有 " << weld_count_this_corner << " 条有效焊缝，尝试强制生成..." << std::endl;
                
                // 强制生成缺失的交线
                for (size_t i = 0; i < corner.plane_ids.size(); i++) {
                    int pid1 = corner.plane_ids[i];
                    int pid2 = corner.plane_ids[(i+1) % corner.plane_ids.size()];
                    auto key = std::make_pair(std::min(pid1, pid2), std::max(pid1, pid2));
                    if (added_seams.find(key) != added_seams.end()) continue;
                    
                    // 扩大采样范围
                    float old_spacing = path_spacing_;
                    path_spacing_ = old_spacing * 2;  // 放宽采样步长
                    
                    WeldSeam seam = createWeldFromIntersection(planes[pid1], planes[pid2], seams.size());
                    path_spacing_ = old_spacing;  // 恢复
                    
                    filterWeldSeamByBounds(seam);
                    
                    if (!seam.path.empty()) {
                        seam.is_corner_weld = true;
                        seams.push_back(seam);
                        added_seams.insert(key);
                        std::cout << "  🔧 强制生成焊缝 " << seam.id << ": 平面 " << pid1 << " - " << pid2 
                                  << ", 路径点 " << seam.path.size() << ", 长度 " << seam.length * 1000 << " mm" << std::endl;
                    }
                }
            }
        }
        std::cout << "边角模式完成，共处理 " << concave_corners << " 个内凹角" << std::endl;
    }
    
    // ========== 2. 长条模式：提取两平面相交的长焊缝 ==========
    if (extraction_mode_ == MODE_LONG_ONLY || extraction_mode_ == MODE_BOTH) {
        std::cout << "\n[长条模式] 提取两平面相交长焊缝..." << std::endl;
        
        for (size_t i = 0; i < planes.size(); i++) {
            for (size_t j = i + 1; j < planes.size(); j++) {
                auto key = std::make_pair(i, j);
                if (added_seams.find(key) != added_seams.end()) continue;
                
                // 检查两个平面的夹角
                float dot = std::abs(planes[i].normal.dot(planes[j].normal));
                float angle = std::acos(dot) * 180.0f / M_PI;
                
                // 只提取夹角在45-135度之间的平面（近似垂直）
                if (angle < 45.0f || angle > 135.0f) continue;
                
                // 检查是否在内凹角范围内（边角模式已处理）
                bool in_corner = false;
                for (const auto& corner : corners) {
                    if (corner.is_concave) {
                        if ((corner.plane_ids[0] == (int)i || corner.plane_ids[1] == (int)i || corner.plane_ids[2] == (int)i) &&
                            (corner.plane_ids[0] == (int)j || corner.plane_ids[1] == (int)j || corner.plane_ids[2] == (int)j)) {
                            in_corner = true;
                            break;
                        }
                    }
                }
                
                // 长条模式：跳过内凹角的边（除非是混合模式）
                if (extraction_mode_ == MODE_LONG_ONLY && in_corner) continue;
                
                // 提取交线
                PlaneExtractor extractor;
                extractor.setDistanceThreshold(0.005f);
                extractor.setMinPlanePoints(100);
                
                CloudPtr line_points = extractor.sampleIntersectionLine(planes[i], planes[j], path_spacing_);
                
                if (line_points->points.size() > 20) {
                    WeldSeam seam = createWeldFromIntersection(planes[i], planes[j], seams.size());
                    filterWeldSeamByBounds(seam);
                    
                    if (!seam.path.empty() && seam.length >= min_weld_length_) {
                        seams.push_back(seam);
                        added_seams.insert(key);
                        std::cout << "  ✅ 长焊缝 " << seam.id << ": 平面 " << i << " - " << j 
                                  << " (夹角 " << angle << "°), 路径点 " << seam.path.size() 
                                  << ", 长度 " << seam.length * 1000 << " mm" << std::endl;
                    }
                }
            }
        }
    }
    
    // 去重：合并端点距离较近的焊缝（可选）
    // ... 可以添加去重逻辑 ...
    
    std::cout << "\n[CornerClassifier] 共提取 " << seams.size() << " 条有效焊缝" << std::endl;
    return seams;
}

// 支持多点验证的版本
bool CornerClassifier::isCameraVisibleToCorner(const CornerPoint& corner,
                                                const std::vector<FinitePlane>& planes,
                                                const Eigen::Vector3f& camera_pos) {
    // ============================================
    // 步骤1：构建局部坐标系
    // ============================================
    // 局部坐标系原点 = 角点位置
    // 三个基向量 = 从角点到三个平面内采样点的单位向量（使用多个点）
    
    // 为每个平面获取多个采样点
    std::vector<std::vector<Eigen::Vector3f>> all_plane_points(3);
    
    // 预定义采样偏移量（在平面内采样多个点）
    std::vector<float> offsets = {-0.03f, 0.0f, 0.03f};  // 采样点偏移（米）
    
    for (size_t idx = 0; idx < corner.plane_ids.size(); idx++) {
        int pid = corner.plane_ids[idx];
        const auto& plane = planes[pid];
        
        // 获取另外两个方向（用于选择采样点）
        std::vector<Eigen::Vector3f> dirs;
        for (int other_pid : corner.plane_ids) {
            if (other_pid == pid) continue;
            const auto& other_plane = planes[other_pid];
            Eigen::Vector3f dir = (other_plane.center - corner.position).normalized();
            dirs.push_back(dir);
        }
        
        // 采样多个点
        std::vector<Eigen::Vector3f> sampled_points;
        
        // 1. 平面中心点
        Eigen::Vector3f center_point = getValidPointInPlane(plane, corner.position, 
                                                             dirs[0], dirs[1]);
        sampled_points.push_back(center_point);
        
        // 2. 平面边界上的多个点
        // 获取平面局部坐标系
        Eigen::Vector3f u = plane.local_x;
        Eigen::Vector3f v = plane.local_y;
        
        // 在平面内采样多个点（围绕中心点）
        for (float du : offsets) {
            for (float dv : offsets) {
                if (du == 0.0f && dv == 0.0f) continue;  // 跳过中心点（已添加）
                
                Eigen::Vector3f candidate = plane.center + u * du + v * dv;
                
                // 检查是否在平面边界内
                PlaneExtractor extractor;
                if (extractor.isPointInPlaneBoundsExact(plane, candidate, 0.01f)) {
                    sampled_points.push_back(candidate);
                }
            }
        }
        
        all_plane_points[idx] = sampled_points;
        
        std::cout << "    [采样] 平面 " << pid << " 采样 " << sampled_points.size() << " 个点" << std::endl;
    }
    
    // ============================================
    // 步骤2：对每组采样点进行可见性验证
    // ============================================
    
    int valid_combs = 0;
    int total_combs = 0;
    
    // 遍历所有组合（每个平面选一个采样点）
    for (size_t i = 0; i < all_plane_points[0].size(); i++) {
        for (size_t j = 0; j < all_plane_points[1].size(); j++) {
            for (size_t k = 0; k < all_plane_points[2].size(); k++) {
                total_combs++;
                
                const auto& p0 = all_plane_points[0][i];
                const auto& p1 = all_plane_points[1][j];
                const auto& p2 = all_plane_points[2][k];
                
                // 构建基向量
                Eigen::Vector3f v0 = (p0 - corner.position).normalized();
                Eigen::Vector3f v1 = (p1 - corner.position).normalized();
                Eigen::Vector3f v2 = (p2 - corner.position).normalized();
                
                // 构建变换矩阵
                Eigen::Matrix3f T;
                T.col(0) = v0;
                T.col(1) = v1;
                T.col(2) = v2;
                
                // 检查矩阵是否可逆
                float det = T.determinant();
                if (std::abs(det) < 1e-4f) continue;
                
                // 将相机位置变换到局部坐标系
                Eigen::Vector3f local_cam = T.inverse() * (camera_pos - corner.position);
                
                // 判断是否所有分量都为正（带容差）
                float threshold = 0.01f;
                bool visible = (local_cam.x() > threshold && 
                                local_cam.y() > threshold && 
                                local_cam.z() > threshold);
                
                if (visible) {
                    valid_combs++;
                }
            }
        }
    }
    
    // ============================================
    // 步骤3：综合判断
    // ============================================
    // 如果有超过50%的采样组合满足条件，则认为相机可以看到这个角点
    float visibility_ratio = (total_combs > 0) ? (float)valid_combs / total_combs : 0.0f;
    bool result = (visibility_ratio > 0.9f);
    
    std::cout << "      [可见性] 有效组合: " << valid_combs << "/" << total_combs 
              << " (" << (visibility_ratio * 100) << "%) → " 
              << (result ? "可见" : "不可见") << std::endl;
    
    return result;
}

// 在 CornerClassifier 类中添加新函数
// 在 CornerClassifier::getValidPointInPlane 函数中，修改调用方式
Eigen::Vector3f CornerClassifier::getValidPointInPlane(const FinitePlane& plane, 
                                                        const Eigen::Vector3f& corner_point,
                                                        const Eigen::Vector3f& other_dir1,
                                                        const Eigen::Vector3f& other_dir2) {
    // 创建一个临时 PlaneExtractor 对象来使用精确边界判断
    PlaneExtractor extractor;
    
    // 首先检查平面中心是否在有限平面内
    if (extractor.isPointInPlaneBoundsExact(plane, plane.center)) {
        // 中心点在平面内，直接使用
        return plane.center;
    }
    
    // 中心点不在平面内，需要寻找一个合适的内部点
    // 策略：从角点出发，沿着平面内的方向采样，直到找到平面内的点
    
    // 获取平面的局部坐标系
    Eigen::Vector3f u = plane.local_x;
    Eigen::Vector3f v = plane.local_y;
    
    // 尝试在平面内搜索一个点，该点位于平面内部且与其他方向尽量正交
    std::vector<Eigen::Vector3f> candidates;
    
    // 从平面边界内均匀采样
    float step = 0.01f;  // 1cm 步长
    float search_range = 0.1f;  // 10cm 范围
    
    for (float du = -search_range; du <= search_range; du += step) {
        for (float dv = -search_range; dv <= search_range; dv += step) {
            Eigen::Vector3f candidate = plane.origin + u * du + v * dv;
            
            // 检查是否在平面边界内
            if (extractor.isPointInPlaneBoundsExact(plane, candidate, 0.005f)) {
                candidates.push_back(candidate);
            }
        }
    }
    
    if (candidates.empty()) {
        // 找不到合适的点，回退到角点沿法线方向偏移
        return corner_point + plane.normal * 0.01f;
    }
    
    // 选择与其他方向向量夹角最接近90度的点
    float best_score = -1;
    Eigen::Vector3f best_point = candidates[0];
    
    for (const auto& p : candidates) {
        Eigen::Vector3f dir = (p - corner_point).normalized();
        
        // 计算与其他方向的夹角
        float dot1 = std::abs(dir.dot(other_dir1));
        float dot2 = std::abs(dir.dot(other_dir2));
        
        // 目标：与两个方向都正交（点积接近0）
        float score = (1 - dot1) + (1 - dot2);
        
        if (score > best_score) {
            best_score = score;
            best_point = p;
        }
    }
    
    std::cout << "    [采样] 中心点不在平面内，使用内部采样点" << std::endl;
    return best_point;
}


// ========== 计算基于可见相机的评分 ==========
float CornerClassifier::computeCornerScoreFromVisibleCameras(const CornerPoint& corner,
                                                              const std::vector<FinitePlane>& planes,
                                                              const std::vector<Eigen::Vector3f>& camera_poses) {
    if (camera_poses.empty()) return 0.0f;
    
    // 为每个平面采样多个点
    std::vector<std::vector<Eigen::Vector3f>> all_plane_points(3);
    std::vector<float> offsets = {-0.02f, 0.0f, 0.02f};  // 采样点偏移
    
    for (size_t idx = 0; idx < corner.plane_ids.size(); idx++) {
        int pid = corner.plane_ids[idx];
        const auto& plane = planes[pid];
        
        // 获取另外两个方向
        std::vector<Eigen::Vector3f> dirs;
        for (int other_pid : corner.plane_ids) {
            if (other_pid == pid) continue;
            const auto& other_plane = planes[other_pid];
            Eigen::Vector3f dir = (other_plane.center - corner.position).normalized();
            dirs.push_back(dir);
        }
        
        std::vector<Eigen::Vector3f> sampled_points;
        
        // 平面中心点
        Eigen::Vector3f center_point = getValidPointInPlane(plane, corner.position, 
                                                             dirs[0], dirs[1]);
        sampled_points.push_back(center_point);
        
        // 平面内的采样点
        Eigen::Vector3f u = plane.local_x;
        Eigen::Vector3f v = plane.local_y;
        PlaneExtractor extractor;
        
        for (float du : offsets) {
            for (float dv : offsets) {
                if (du == 0.0f && dv == 0.0f) continue;
                
                Eigen::Vector3f candidate = plane.center + u * du + v * dv;
                if (extractor.isPointInPlaneBoundsExact(plane, candidate, 0.01f)) {
                    sampled_points.push_back(candidate);
                }
            }
        }
        
        all_plane_points[idx] = sampled_points;
    }
    
    // 计算多个采样组合的平均评分
    float total_score = 0.0f;
    int valid_cameras = 0;
    
    for (const auto& cam_pos : camera_poses) {
        // ✅ 检查相机是否可以看到这个角点（使用严格的多点验证）
        if (!isCameraVisibleToCorner(corner, planes, cam_pos)) {
            continue;
        }
        
        valid_cameras++;
        
        // 对每个采样组合计算评分
        float comb_score_sum = 0.0f;
        int comb_count = 0;
        
        for (size_t i = 0; i < all_plane_points[0].size(); i++) {
            for (size_t j = 0; j < all_plane_points[1].size(); j++) {
                for (size_t k = 0; k < all_plane_points[2].size(); k++) {
                    comb_count++;
                    
                    const auto& p0 = all_plane_points[0][i];
                    const auto& p1 = all_plane_points[1][j];
                    const auto& p2 = all_plane_points[2][k];
                    
                    // 计算三个采样点的向量和
                    Eigen::Vector3f v0 = (p0 - corner.position).normalized();
                    Eigen::Vector3f v1 = (p1 - corner.position).normalized();
                    Eigen::Vector3f v2 = (p2 - corner.position).normalized();
                    
                    Eigen::Vector3f sum_vec = v0 + v1 + v2;
                    sum_vec.normalize();
                    
                    // 计算从角点到相机的向量
                    Eigen::Vector3f to_camera = (cam_pos - corner.position);
                    if (to_camera.norm() < 1e-6f) continue;
                    to_camera.normalize();
                    
                    // 计算夹角相似度
                    float dot = sum_vec.dot(to_camera);
                    dot = std::max(-1.0f, std::min(1.0f, dot));
                    
                    comb_score_sum += dot;
                }
            }
        }
        
        if (comb_count > 0) {
            total_score += comb_score_sum / comb_count;
        }
    }
    
    // ✅ 要求至少有3个相机可见
    const int min_visible_cameras = 3;
    if (valid_cameras < min_visible_cameras) {
        std::cout << "    [评分] 可见相机数=" << valid_cameras 
                  << " < " << min_visible_cameras << "，判定为无效" << std::endl;
        return 0.0f;
    }
    
    float avg_score = total_score / valid_cameras;
    
    std::cout << "    [评分] 有效相机数=" << valid_cameras 
              << " (≥" << min_visible_cameras << "), 平均评分=" << avg_score << std::endl;
    
    return avg_score;
}

// ========== 保存角点信息到文件 ==========
void CornerClassifier::saveCornersToFile(const std::vector<CornerPoint>& corners, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[CornerClassifier] 无法保存角点信息: " << filename << std::endl;
        return;
    }
    
    file << "id,x,y,z,is_concave,confidence,plane0,plane1,plane2" << std::endl;
    
    for (size_t i = 0; i < corners.size(); i++) {
        const auto& c = corners[i];
        file << i << ","
             << std::fixed << std::setprecision(6)
             << c.position.x() << "," << c.position.y() << "," << c.position.z() << ","
             << (c.is_concave ? 1 : 0) << "," << c.confidence << ","
             << c.plane_ids[0] << "," << c.plane_ids[1] << "," << c.plane_ids[2] << std::endl;
    }
    
    file.close();
    std::cout << "[CornerClassifier] 保存角点信息: " << filename << std::endl;
}


std::vector<CornerPoint> CornerClassifier::classifyCorners(
    const std::vector<FinitePlane>& planes,
    const std::vector<Eigen::Vector3f>& intersection_points,
    const std::vector<std::tuple<int, int, int>>& intersection_plane_indices) {
    
    std::vector<CornerPoint> corners;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "[CornerClassifier] 角点分类" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "总角点数: " << intersection_points.size() << std::endl;
    std::cout << "总相机数: " << camera_poses_.size() << std::endl;
    
    // 输出边界过滤状态
    std::cout << "边界过滤: " << (filter_boundary_corners_ ? "启用" : "禁用") << std::endl;
    if (filter_boundary_corners_) {
        std::cout << "边界距离阈值: " << boundary_margin_mm_ << " mm" << std::endl;
    }
    
    // 输出模型包围盒信息
    std::cout << "模型包围盒: X[" << cloud_bbox_min_.x() << ", " << cloud_bbox_max_.x() << "]" << std::endl;
    std::cout << "            Y[" << cloud_bbox_min_.y() << ", " << cloud_bbox_max_.y() << "]" << std::endl;
    std::cout << "            Z[" << cloud_bbox_min_.z() << ", " << cloud_bbox_max_.z() << "]" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int concave_count = 0;
    int convex_count = 0;
    int boundary_filtered_count = 0;
    
    for (size_t idx = 0; idx < intersection_points.size(); idx++) {
        const auto& pt = intersection_points[idx];
        auto [i, j, k] = intersection_plane_indices[idx];
        
        // 验证索引有效性
        if (i >= (int)planes.size() || j >= (int)planes.size() || k >= (int)planes.size()) {
            std::cerr << "[错误] 索引超出范围: " << i << "," << j << "," << k << std::endl;
            continue;
        }
        
        CornerPoint corner;
        corner.position = pt;
        corner.plane_ids = {i, j, k};
        
        std::cout << "\n[角点 " << idx << "] " << std::endl;
        std::cout << "  位置: (" << std::fixed << std::setprecision(4)
                  << pt.x() << ", " << pt.y() << ", " << pt.z() << ")" << std::endl;
        std::cout << "  平面: (" << i << "," << j << "," << k << ")" << std::endl;
        
        // 边界过滤检查
        bool inside_valid_region = true;
        if (filter_boundary_corners_) {
            inside_valid_region = isCornerInsideModel(pt, boundary_margin_mm_);
            float dist_to_boundary = distanceToNearestBoundary(pt);
            std::cout << "  到最近边界距离: " << dist_to_boundary * 1000 << " mm" << std::endl;
        }
        
        if (!inside_valid_region) {
            corner.is_concave = false;
            convex_count++;
            boundary_filtered_count++;
            corner.confidence = 0.0f;
            
            // 计算外凸方向
            Eigen::Vector3f sum_center(0, 0, 0);
            for (int pid : corner.plane_ids) {
                sum_center += planes[pid].center;
            }
            corner.inward_direction = (sum_center / 3 - corner.position).normalized();
            
            corners.push_back(corner);
            std::cout << "  ❌ 判定结果: 外凸点 (边界区域被过滤)" << std::endl;
            continue;
        }
        
        // 计算三条焊缝
        std::vector<WeldSegment> segments(3);
        std::vector<std::pair<int, int>> plane_pairs = {
            {i, j}, {j, k}, {k, i}
        };
        
        bool valid_welds = true;
        for (int s = 0; s < 3; s++) {
            int pid1 = plane_pairs[s].first;
            int pid2 = plane_pairs[s].second;
            
            segments[s].plane_id1 = pid1;
            segments[s].plane_id2 = pid2;
            segments[s].start_point = pt;
            segments[s].end_point = computeWeldEndPoint(planes[pid1], planes[pid2], pt);
            
            segments[s].length = (segments[s].end_point - pt).norm();
            if (segments[s].length < 0.005f) {
                valid_welds = false;
                break;
            }
        }
        
        if (!valid_welds) {
            corner.is_concave = false;
            convex_count++;
            std::cout << "  ❌ 判定结果: 外凸点 (焊缝长度过短)" << std::endl;
            corners.push_back(corner);
            continue;
        }
        
        // 使用焊缝向量验证
        bool is_concave = isConcaveWithWeldVectors(corner, planes, camera_poses_, segments);
        
        corner.is_concave = is_concave;
        corner.confidence = is_concave ? 0.9f : 0.1f;
        
        // 计算内凹方向
        Eigen::Vector3f sum_dirs(0, 0, 0);
        for (const auto& seg : segments) {
            sum_dirs += (seg.end_point - pt).normalized();
        }
        corner.inward_direction = sum_dirs.normalized();
        
        corners.push_back(corner);
        
        if (is_concave) {
            concave_count++;
            std::cout << "  ✅ 判定结果: 内凹角 (焊缝) - 焊缝长度: ["
                      << segments[0].length * 1000 << ", "
                      << segments[1].length * 1000 << ", "
                      << segments[2].length * 1000 << "] mm" << std::endl;
        } else {
            convex_count++;
            std::cout << "  ❌ 判定结果: 外凸角 (边角)" << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "[角点统计]" << std::endl;
    std::cout << "  内凹(焊缝): " << concave_count << std::endl;
    std::cout << "  外凸(边角): " << convex_count << std::endl;
    if (filter_boundary_corners_) {
        std::cout << "  边界过滤: " << boundary_filtered_count << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    saveCornersToFile(corners, "corners_info.txt");
    
    return corners;
}


// ========== 统计有多少相机可以看到这个内凹角 ==========
int CornerClassifier::countCamerasSeeingConcaveCorner(const CornerPoint& corner,
                                                       const std::vector<FinitePlane>& planes,
                                                       const std::vector<Eigen::Vector3f>& camera_poses,
                                                       bool use_multi_sample) {
    int visible_count = 0;
    int total_cameras = camera_poses.size();
    
    std::cout << "    [相机可见性检查] 共 " << total_cameras << " 个相机" << std::endl;
    
    for (size_t cam_idx = 0; cam_idx < camera_poses.size(); cam_idx++) {
        const auto& cam_pos = camera_poses[cam_idx];
        
        std::cout << "      相机 " << cam_idx << ": ";
        
        bool visible = isCameraInsideCornerMultiSample(corner, planes, cam_pos);
        
        if (visible) {
            visible_count++;
            std::cout << " ✓ 可见" << std::endl;
        } else {
            std::cout << " ✗ 不可见" << std::endl;
        }
    }
    
    std::cout << "    [统计] 可见相机数: " << visible_count << "/" << total_cameras << std::endl;
    
    return visible_count;
}


// ========== 多点采样验证：每个平面采样多个点，所有组合都必须满足条件 ==========
// ========== 改进版：从点云中智能采样（避免重复、确保边界内）==========
// ========== 唯一判定函数：从点云中随机采样，所有组合都必须满足条件 ==========
bool CornerClassifier::isCameraInsideCornerMultiSample(const CornerPoint& corner,
                                                        const std::vector<FinitePlane>& planes,
                                                        const Eigen::Vector3f& camera_pos) {
    // 采样参数：每个平面采样多少个点
    int num_samples_per_plane = 8;  // 每个平面采样8个点，增加采样密度
    
    // 为每个平面收集多个采样点（只从点云中采样）
    std::vector<std::vector<Eigen::Vector3f>> all_plane_samples(3);
    
    for (size_t idx = 0; idx < corner.plane_ids.size(); idx++) {
        int pid = corner.plane_ids[idx];
        const auto& plane = planes[pid];
        
        if (!plane.cloud || plane.cloud->points.empty()) {
            std::cout << "      [错误] 平面 " << pid << " 点云为空，无法采样" << std::endl;
            return false;
        }
        
        const auto& cloud_points = plane.cloud->points;
        int total_points = cloud_points.size();
        
        // 随机采样不重复的点
        std::vector<Eigen::Vector3f> sampled_points;
        std::set<int> used_indices;
        
        // 初始化随机种子
        static bool seed_initialized = false;
        if (!seed_initialized) {
            std::srand(static_cast<unsigned int>(std::time(nullptr)));
            seed_initialized = true;
        }
        
        // 采样 num_samples_per_plane 个不重复的点
        int samples_to_take = std::min(num_samples_per_plane, total_points);
        while (sampled_points.size() < (size_t)samples_to_take && used_indices.size() < (size_t)total_points) {
            int random_idx = std::rand() % total_points;
            if (used_indices.find(random_idx) != used_indices.end()) {
                continue;
            }
            used_indices.insert(random_idx);
            
            const auto& p = cloud_points[random_idx];
            sampled_points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
        }
        
        if (sampled_points.empty()) {
            std::cout << "      [错误] 平面 " << pid << " 无法采样到有效点" << std::endl;
            return false;
        }
        
        all_plane_samples[idx] = sampled_points;
        
        std::cout << "      [采样] 平面 " << pid << " 从 " << total_points 
                  << " 个点中采样 " << sampled_points.size() << " 个点" << std::endl;
    }
    
    // 确保每个平面至少有1个采样点
    for (int i = 0; i < 3; i++) {
        if (all_plane_samples[i].empty()) {
            std::cout << "      [错误] 平面 " << corner.plane_ids[i] << " 没有有效采样点" << std::endl;
            return false;
        }
    }
    
    // 遍历所有采样组合（每个平面选一个点）
    int total_combs = 0;
    int valid_combs = 0;
    int invalid_matrix_combs = 0;
    
    for (size_t i = 0; i < all_plane_samples[0].size(); i++) {
        for (size_t j = 0; j < all_plane_samples[1].size(); j++) {
            for (size_t k = 0; k < all_plane_samples[2].size(); k++) {
                total_combs++;
                
                const auto& p0 = all_plane_samples[0][i];
                const auto& p1 = all_plane_samples[1][j];
                const auto& p2 = all_plane_samples[2][k];
                
                Eigen::Vector3f dir0 = (p0 - corner.position).normalized();
                Eigen::Vector3f dir1 = (p1 - corner.position).normalized();
                Eigen::Vector3f dir2 = (p2 - corner.position).normalized();
                
                // 构建变换矩阵
                Eigen::Matrix3f T;
                T.col(0) = dir0;
                T.col(1) = dir1;
                T.col(2) = dir2;
                
                // 检查矩阵是否可逆
                float det = T.determinant();
                if (std::abs(det) < 1e-6f) {
                    invalid_matrix_combs++;
                    continue;
                }
                
                // 变换相机位置到局部坐标系
                Eigen::Vector3f local_cam = T.inverse() * (camera_pos - corner.position);
                
                // 严格要求：三个分量都必须大于阈值
                float threshold = 0.01f;  // 防止边缘点入侵
                if (local_cam.x() > threshold && 
                    local_cam.y() > threshold && 
                    local_cam.z() > threshold) {
                    valid_combs++;
                }
            }
        }
    }
    
    int valid_total_combs = total_combs - invalid_matrix_combs;
    
    // 严格要求：所有有效组合都必须满足条件（100%）
    bool all_valid = (valid_total_combs > 0) && (valid_combs == valid_total_combs);
    
    float valid_percent = (valid_total_combs > 0) ? (100.0f * valid_combs / valid_total_combs) : 0.0f;
    std::cout << "      [可见性判定] 总组合=" << total_combs 
              << ", 有效组合=" << valid_total_combs
              << ", 满足条件=" << valid_combs
              << " (" << std::fixed << std::setprecision(1) << valid_percent << "%) → "
              << (all_valid ? "✓ 相机可见（内凹角）" : "✗ 相机不可见（外凸角）") << std::endl;
    
    return all_valid;
}

// 在 CornerClassifier 类中添加边界过滤函数
// 检查角点是否在模型内部（距离边界有一定距离）
bool CornerClassifier::isCornerInsideModel(const Eigen::Vector3f& corner, float margin_mm) {
    if (!cloud_ || cloud_->points.empty()) return true;
    
    float margin = margin_mm / 1000.0f;  // 转换为米
    
    // 检查角点是否离包围盒边界太近
    bool too_close_to_boundary = 
        (corner.x() - cloud_bbox_min_.x() < margin) ||
        (cloud_bbox_max_.x() - corner.x() < margin) ||
        (corner.y() - cloud_bbox_min_.y() < margin) ||
        (cloud_bbox_max_.y() - corner.y() < margin) ||
        (corner.z() - cloud_bbox_min_.z() < margin) ||
        (cloud_bbox_max_.z() - corner.z() < margin);
    
    if (too_close_to_boundary) {
        std::cout << "  [边界过滤] 角点距离边界太近（< " << margin_mm << "mm），判定为外凸点" << std::endl;
        return false;
    }
    
    return true;
}

// 计算角点到最近边界的距离
float CornerClassifier::distanceToNearestBoundary(const Eigen::Vector3f& corner) {
    float dx = std::min(corner.x() - cloud_bbox_min_.x(), cloud_bbox_max_.x() - corner.x());
    float dy = std::min(corner.y() - cloud_bbox_min_.y(), cloud_bbox_max_.y() - corner.y());
    float dz = std::min(corner.z() - cloud_bbox_min_.z(), cloud_bbox_max_.z() - corner.z());
    return std::min({dx, dy, dz});
}

// // ========== 核心函数：检查相机是否从角点内部可见 ==========
// // 以角点为原点，三个平面中心方向为基，检查相机坐标是否三个分量都为正
// bool CornerClassifier::isCameraInsideCorner(const CornerPoint& corner,
//                                              const std::vector<FinitePlane>& planes,
//                                              const Eigen::Vector3f& camera_pos) {
//     // ============================================
//     // 步骤1：为每个平面获取多个采样点
//     // ============================================
//     std::vector<Eigen::Vector3f> plane_centers;
    
//     for (int pid : corner.plane_ids) {
//         const auto& plane = planes[pid];
        
//         // 获取另外两个平面方向（用于选择采样方向）
//         std::vector<Eigen::Vector3f> other_dirs;
//         for (int other_pid : corner.plane_ids) {
//             if (other_pid == pid) continue;
//             const auto& other_plane = planes[other_pid];
//             Eigen::Vector3f dir = (other_plane.center - corner.position).normalized();
//             other_dirs.push_back(dir);
//         }
        
//         // 获取一个有效的平面内点（不一定是中心）
//         Eigen::Vector3f point_in_plane = getValidPointInPlane(plane, corner.position, 
//                                                                other_dirs[0], other_dirs[1]);
//         plane_centers.push_back(point_in_plane);
//     }
    
//     // ============================================
//     // 步骤2：构建局部坐标系（三个方向向量）
//     // ============================================
//     Eigen::Vector3f dir0 = (plane_centers[0] - corner.position).normalized();
//     Eigen::Vector3f dir1 = (plane_centers[1] - corner.position).normalized();
//     Eigen::Vector3f dir2 = (plane_centers[2] - corner.position).normalized();
    
//     // 构建变换矩阵 T（列向量为基向量）
//     Eigen::Matrix3f T;
//     T.col(0) = dir0;
//     T.col(1) = dir1;
//     T.col(2) = dir2;
    
//     // 检查矩阵是否可逆
//     float det = T.determinant();
//     if (std::abs(det) < 1e-6f) {
//         std::cout << "      [错误] 矩阵不可逆，三个方向共面" << std::endl;
//         return false;
//     }
    
//     // ============================================
//     // 步骤3：将相机位置变换到局部坐标系
//     // ============================================
//     Eigen::Vector3f local_cam = T.inverse() * (camera_pos - corner.position);
    
//     // ============================================
//     // 步骤4：检查是否三个分量都为正（内凹角可见性）
//     // ============================================
//     float threshold = 0.01f;  // 防止边缘点入侵
//     bool all_positive = (local_cam.x() > threshold && 
//                          local_cam.y() > threshold && 
//                          local_cam.z() > threshold);
    
//     if (all_positive) {
//         std::cout << "      [可见] 局部坐标: ("
//                   << std::fixed << std::setprecision(4)
//                   << local_cam.x() << ", " << local_cam.y() << ", " << local_cam.z() << ")" << std::endl;
//     }
    
//     return all_positive;
// }


// 计算从角点出发的焊缝终点（交线的最远点）
Eigen::Vector3f CornerClassifier::computeWeldEndPoint(const FinitePlane& p1, 
                                                        const FinitePlane& p2,
                                                        const Eigen::Vector3f& corner_point) {
    // 计算两平面交线
    Eigen::Vector3f direction, point_on_line;
    PlaneExtractor extractor;
    extractor.computePlaneIntersectionLine(p1, p2, direction, point_on_line);
    
    // 将角点投影到交线上，得到焊缝起点参数
    float t_corner = (corner_point - point_on_line).dot(direction);
    
    // 确定交线的有效范围（在两个平面的边界内）
    std::vector<float> t_values;
    
    // 采样平面的边界点，投影到交线上
    auto addBoundaryPoints = [&](const FinitePlane& plane) {
        std::vector<Eigen::Vector3f> boundary_pts;
        boundary_pts.push_back(Eigen::Vector3f(plane.min_x, plane.min_y, plane.min_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.min_x, plane.min_y, plane.max_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.min_x, plane.max_y, plane.min_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.min_x, plane.max_y, plane.max_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.max_x, plane.min_y, plane.min_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.max_x, plane.min_y, plane.max_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.max_x, plane.max_y, plane.min_z));
        boundary_pts.push_back(Eigen::Vector3f(plane.max_x, plane.max_y, plane.max_z));
        
        for (const auto& pt : boundary_pts) {
            float t = (pt - point_on_line).dot(direction);
            t_values.push_back(t);
        }
    };
    
    addBoundaryPoints(p1);
    addBoundaryPoints(p2);
    
    if (t_values.empty()) return corner_point;
    
    // 找到离角点最远的有效端点
    float t_min = *std::min_element(t_values.begin(), t_values.end());
    float t_max = *std::max_element(t_values.begin(), t_values.end());
    
    // 选择离角点最远的端点
    float dist_to_min = std::abs(t_corner - t_min);
    float dist_to_max = std::abs(t_corner - t_max);
    
    float t_end = (dist_to_min > dist_to_max) ? t_min : t_max;
    Eigen::Vector3f end_point = point_on_line + t_end * direction;
    
    return end_point;
}

// 使用三条焊缝向量判定内凹角
bool CornerClassifier::isConcaveWithWeldVectors(const CornerPoint& corner,
                                                 const std::vector<FinitePlane>& planes,
                                                 const std::vector<Eigen::Vector3f>& camera_poses,
                                                 const std::vector<WeldSegment>& weld_segments) {
    
    if (weld_segments.size() != 3) return false;
    
    // 步骤1：使用三条焊缝向量构建局部坐标系
    Eigen::Vector3f v0 = (weld_segments[0].end_point - corner.position).normalized();
    Eigen::Vector3f v1 = (weld_segments[1].end_point - corner.position).normalized();
    Eigen::Vector3f v2 = (weld_segments[2].end_point - corner.position).normalized();
    
    Eigen::Matrix3f T;
    T.col(0) = v0;
    T.col(1) = v1;
    T.col(2) = v2;
    
    if (std::abs(T.determinant()) < 1e-6f) return false;
    
    // 步骤2：统计满足条件的相机
    int valid_cameras = 0;
    
    for (const auto& cam_pos : camera_poses) {
        Eigen::Vector3f local_cam = T.inverse() * (cam_pos - corner.position);
        
        float threshold = 0.01f;
        bool all_positive = (local_cam.x() > threshold && 
                             local_cam.y() > threshold && 
                             local_cam.z() > threshold);
        
        if (!all_positive) continue;
        
        // 步骤3：验证焊缝终点距离条件
        bool distance_condition = true;
        for (const auto& seg : weld_segments) {
            // 相机到焊缝终点的距离
            float dist_to_end = (cam_pos - seg.end_point).norm();
            
            // 在相交的两个平面上采样点
            const auto& plane1 = planes[seg.plane_id1];
            const auto& plane2 = planes[seg.plane_id2];
            
            // 检查相机到平面上任意点的距离是否都小于到焊缝终点的距离
            // 简化：检查到平面中心的距离
            float dist_to_plane1 = (cam_pos - plane1.center).norm();
            float dist_to_plane2 = (cam_pos - plane2.center).norm();
            
            if (dist_to_end < dist_to_plane1 * 0.8f || dist_to_end < dist_to_plane2 * 0.8f) {
                distance_condition = false;
                break;
            }
        }
        
        if (distance_condition) {
            valid_cameras++;
        }
    }
    
    // 至少2个相机满足条件才认为是内凹角
    return valid_cameras >= 2;
}

// 计算收缩后的包围盒
void CornerClassifier::computeShrunkBBox(float shrink_margin_mm) {
    float shrink = shrink_margin_mm / 1000.0f;  // 转换为米
    
    shrunk_bbox_min_ = cloud_bbox_min_ + Eigen::Vector3f(shrink, shrink, shrink);
    shrunk_bbox_max_ = cloud_bbox_max_ - Eigen::Vector3f(shrink, shrink, shrink);
    
    std::cout << "[包围盒] 原始: [" << cloud_bbox_min_.x() << ", " << cloud_bbox_max_.x() << "]"
              << " x [" << cloud_bbox_min_.y() << ", " << cloud_bbox_max_.y() << "]"
              << " x [" << cloud_bbox_min_.z() << ", " << cloud_bbox_max_.z() << "]" << std::endl;
    std::cout << "[包围盒] 收缩" << shrink_margin_mm << "mm后: ["
              << shrunk_bbox_min_.x() << ", " << shrunk_bbox_max_.x() << "]"
              << " x [" << shrunk_bbox_min_.y() << ", " << shrunk_bbox_max_.y() << "]"
              << " x [" << shrunk_bbox_min_.z() << ", " << shrunk_bbox_max_.z() << "]" << std::endl;
}
