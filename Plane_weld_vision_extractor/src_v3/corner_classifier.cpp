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

CornerClassifier::CornerClassifier() : cloud_(new Cloud) {}

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

// CornerClassifier::classifyCorners(v1): 综合分类函数：结合多种方法判断角点类型，并计算置信度
// std::vector<CornerPoint> CornerClassifier::classifyCorners(const std::vector<FinitePlane>& planes,
//                                                             const std::vector<Eigen::Vector3f>& intersection_points) {
//     std::vector<CornerPoint> corners;
    
//     std::cout << "[CornerClassifier] 开始分类 " << intersection_points.size() << " 个角点" << std::endl;
    
//     for (const auto& pt : intersection_points) {
//         // 找到包含该点的三个平面
//         std::vector<std::pair<float, int>> distances;
//         for (size_t i = 0; i < planes.size(); i++) {
//             float dist = std::abs(planes[i].normal.dot(pt) + planes[i].d);
//             distances.push_back({dist, i});
//         }
        
//         std::sort(distances.begin(), distances.end());
        
//         if (distances.size() >= 3 && distances[0].first < 0.01f && 
//             distances[1].first < 0.01f && distances[2].first < 0.01f) {
            
//             CornerPoint corner;
//             corner.position = pt;
//             corner.plane_ids = {distances[0].second, distances[1].second, distances[2].second};
            
//             // 多方法综合判断
//             float score = 0;
//             int method_count = 1;
            
//             bool vec_sum = classifyByVectorSum(corner, planes);
//             score += vec_sum ? 1 : -1;
            
//             if (!camera_poses_.empty()) {
//                 bool camera = classifyByCameraPose(corner, planes);
//                 score += camera ? 1 : -1;
//                 method_count++;
//             }
            
//             bool normal_dist = classifyByNormalDistribution(corner, planes);
//             score += normal_dist ? 1 : -1;
//             method_count++;
            
//             corner.confidence = std::abs(score) / method_count;
//             corner.is_concave = score > 0;
            
//             // 计算内凹方向
//             if (corner.is_concave) {
//                 corner.inward_direction = -vectorToPlaneCenter(corner, planes[corner.plane_ids[0]]);
//             } else {
//                 corner.inward_direction = vectorToPlaneCenter(corner, planes[corner.plane_ids[0]]);
//             }
//             corner.inward_direction.normalize();
            
//             corners.push_back(corner);
            
//             std::cout << "[角点] 位置: (" << pt.x() << ", " << pt.y() << ", " << pt.z() << ")"
//                       << ", 内凹: " << (corner.is_concave ? "是(焊缝)" : "否(边角)")
//                       << ", 置信度: " << corner.confidence << std::endl;
//         }
//     }
    
//     return corners;
// }


// CornerClassifier::classifyCorners 函数(v2)，使用新的评分方法：相机可见性 + 向量夹角评分法，增强分类效果，并输出更详细的日志信息
std::vector<CornerPoint> CornerClassifier::classifyCorners(
    const std::vector<FinitePlane>& planes,
    const std::vector<Eigen::Vector3f>& intersection_points,
    const std::vector<std::tuple<int, int, int>>& intersection_plane_indices) {
    
    std::vector<CornerPoint> corners;
    
    std::cout << "[CornerClassifier] 开始分类 " << intersection_points.size() << " 个角点" << std::endl;
    std::cout << "[CornerClassifier] 使用相机可见性 + 向量夹角评分法" << std::endl;
    
    float score_threshold = 0.3f;  // 评分阈值
    
    for (size_t idx = 0; idx < intersection_points.size(); idx++) {
        const auto& pt = intersection_points[idx];
        auto [i, j, k] = intersection_plane_indices[idx];
        
        // 验证索引有效性
        if (i >= (int)planes.size() || j >= (int)planes.size() || k >= (int)planes.size()) {
            std::cerr << "[错误] 索引超出范围: " << i << "," << j << "," << k 
                      << " 平面总数=" << planes.size() << std::endl;
            continue;
        }
        
        CornerPoint corner;
        corner.position = pt;
        corner.plane_ids = {i, j, k};  // 直接使用传入的索引
        
        // 使用原来的评分方法（不变）
        float score = computeCornerScoreFromVisibleCameras(corner, planes, camera_poses_);
        
        corner.confidence = std::abs(score);
        corner.is_concave = score > score_threshold;
        
        // 计算内凹方向
        if (corner.is_concave) {
            corner.inward_direction = -vectorToPlaneCenter(corner, planes[corner.plane_ids[0]]);
        } else {
            corner.inward_direction = vectorToPlaneCenter(corner, planes[corner.plane_ids[0]]);
        }
        corner.inward_direction.normalize();
        
        corners.push_back(corner);
        
        std::cout << "[角点] 位置: (" << std::fixed << std::setprecision(4)
                  << pt.x() << ", " << pt.y() << ", " << pt.z() << ")"
                  << ", 平面索引: (" << i << "," << j << "," << k << ")"
                  << ", 评分: " << score
                  << ", 内凹: " << (corner.is_concave ? "是(焊缝)" : "否(边角)")
                  << ", 置信度: " << corner.confidence << std::endl;
    }
    
    // 保存角点信息到文件
    saveCornersToFile(corners, "corners_info.txt");
    
    return corners;
}

// ========== 获取指向外部的法向量 ==========
// Eigen::Vector3f CornerClassifier::getOutwardNormal(const FinitePlane& plane, 
//                                                     const Eigen::Vector3f& corner_point) {
//     // 计算从角点到平面中心的向量
//     Eigen::Vector3f to_center = plane.center - corner_point;
    
//     // 如果法向量与 to_center 方向一致，则指向外部；否则反转
//     if (plane.normal.dot(to_center) > 0) {
//         return plane.normal;
//     } else {
//         return -plane.normal;
//     }
// }

// // ========== 通用内凹角判断 ==========
// bool CornerClassifier::isConcaveCornerGeneral(const CornerPoint& corner,
//                                                 const std::vector<FinitePlane>& planes) {
//     // 获取三个平面的外法线（指向角外部）
//     std::vector<Eigen::Vector3f> outward_normals;
//     for (int pid : corner.plane_ids) {
//         const auto& plane = planes[pid];
//         Eigen::Vector3f to_center = plane.center - corner.position;
//         if (plane.normal.dot(to_center) > 0) {
//             outward_normals.push_back(plane.normal);
//         } else {
//             outward_normals.push_back(-plane.normal);
//         }
//     }
    
//     // 计算三个外法线的和向量
//     Eigen::Vector3f sum_normals = outward_normals[0] + outward_normals[1] + outward_normals[2];
//     float sum_magnitude = sum_normals.norm();
    
//     // 如果和向量很小（相互抵消），说明法线指向不同方向 → 内凹角
//     // 如果和向量很大（方向一致），说明法线指向同一方向 → 外凸角
//     bool is_concave = (sum_magnitude < 1.5f);
    
//     std::cout << "    [通用判断] 外法线和大小: " << sum_magnitude 
//               << " → " << (is_concave ? "内凹角(焊缝)" : "外凸角(边角)") << std::endl;
    
//     return is_concave;
// }


// // ========== 改进的角点分类函数 ==========
// std::vector<CornerPoint> CornerClassifier::classifyCorners(const std::vector<FinitePlane>& planes,
//                                                             const std::vector<Eigen::Vector3f>& intersection_points) {
//     std::vector<CornerPoint> corners;
    
//     std::cout << "[CornerClassifier] 开始分类 " << intersection_points.size() << " 个角点" << std::endl;
//     std::cout << "[CornerClassifier] 使用通用数学方法：外法线和判断法" << std::endl;
    
//     for (const auto& pt : intersection_points) {
//         // 找到包含该点的三个平面
//         std::vector<std::pair<float, int>> distances;
//         for (size_t i = 0; i < planes.size(); i++) {
//             float dist = std::abs(planes[i].normal.dot(pt) + planes[i].d);
//             distances.push_back({dist, i});
//         }
        
//         std::sort(distances.begin(), distances.end());
        
//         if (distances.size() >= 3 && distances[0].first < 0.02f && 
//             distances[1].first < 0.02f && distances[2].first < 0.02f) {
            
//             CornerPoint corner;
//             corner.position = pt;
//             corner.plane_ids = {distances[0].second, distances[1].second, distances[2].second};
            
//             // ✅ 使用通用数学方法判断
//             corner.is_concave = isConcaveCornerGeneral(corner, planes);
//             corner.confidence = corner.is_concave ? 1.0f : 1.0f;
            
//             // 计算内凹方向
//             if (corner.is_concave) {
//                 corner.inward_direction = -vectorToPlaneCenter(corner, planes[corner.plane_ids[0]]);
//             } else {
//                 corner.inward_direction = vectorToPlaneCenter(corner, planes[corner.plane_ids[0]]);
//             }
//             corner.inward_direction.normalize();
            
//             corners.push_back(corner);
            
//             std::cout << "[角点] 位置: (" << std::fixed << std::setprecision(4)
//                       << pt.x() << ", " << pt.y() << ", " << pt.z() << ")"
//                       << ", 内凹: " << (corner.is_concave ? "是(焊缝)" : "否(边角)")
//                       << std::endl;
//         }
//     }
    
//     int concave_count = std::count_if(corners.begin(), corners.end(), 
//                                        [](const CornerPoint& c) { return c.is_concave; });
//     std::cout << "[CornerClassifier] 角点分类完成：内凹(焊缝)=" << concave_count 
//               << ", 外凸(边角)=" << (corners.size() - concave_count) << std::endl;
    
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



// 焊缝交线提取v1: 全部提取（角焊缝 + 长焊缝）
// std::vector<WeldSeam> CornerClassifier::extractWeldSeams(const std::vector<FinitePlane>& planes,
//                                                           const std::vector<CornerPoint>& corners) {
//     std::vector<WeldSeam> seams;
//     std::set<std::pair<int, int>> added_seams;
    
//     // 1. 从内凹角提取焊缝（三平面交线）
//     for (const auto& corner : corners) {
//         if (!corner.is_concave) continue;
        
//         std::cout << "[焊缝] 处理内凹角: 平面 ";
//         for (int pid : corner.plane_ids) std::cout << pid << " ";
//         std::cout << std::endl;
        
//         // 提取三条交线
//         for (size_t i = 0; i < corner.plane_ids.size(); i++) {
//             int pid1 = corner.plane_ids[i];
//             int pid2 = corner.plane_ids[(i+1) % corner.plane_ids.size()];
            
//             auto key = std::make_pair(std::min(pid1, pid2), std::max(pid1, pid2));
//             if (added_seams.find(key) != added_seams.end()) continue;
            
//             WeldSeam seam = createWeldFromIntersection(planes[pid1], planes[pid2], seams.size());
//             if (!seam.path.empty()) {
//                 seam.is_corner_weld = true;
//                 seams.push_back(seam);
//                 added_seams.insert(key);
//                 std::cout << "  + 焊缝 " << seam.id << ": 平面 " << pid1 << " - " << pid2 
//                           << ", 路径点 " << seam.path.size() << ", 长度 " << seam.length * 1000 << " mm" << std::endl;
//             }
//         }
//     }
    
//     // 2. 提取长焊缝（两平面相交，不在内凹角范围内）
//     for (size_t i = 0; i < planes.size(); i++) {
//         for (size_t j = i + 1; j < planes.size(); j++) {
//             auto key = std::make_pair(i, j);
//             if (added_seams.find(key) != added_seams.end()) continue;
            
//             // 检查是否在同一内凹角中
//             bool in_corner = false;
//             for (const auto& corner : corners) {
//                 if (corner.is_concave) {
//                     if ((corner.plane_ids[0] == i || corner.plane_ids[1] == i || corner.plane_ids[2] == i) &&
//                         (corner.plane_ids[0] == j || corner.plane_ids[1] == j || corner.plane_ids[2] == j)) {
//                         in_corner = true;
//                         break;
//                     }
//                 }
//             }
            
//             if (in_corner) continue;
            
//             // 检查交线长度
//             PlaneExtractor extractor;
//             CloudPtr line_points = extractor.sampleIntersectionLine(planes[i], planes[j], path_spacing_);
            
//             if (line_points->points.size() > 20) {
//                 WeldSeam seam = createWeldFromIntersection(planes[i], planes[j], seams.size());
//                 if (!seam.path.empty()) {
//                     seams.push_back(seam);
//                     added_seams.insert(key);
//                     std::cout << "[焊缝] 长焊缝 " << seam.id << ": 平面 " << i << " - " << j 
//                               << ", 路径点 " << seam.path.size() << ", 长度 " << seam.length * 1000 << " mm" << std::endl;
//                 }
//             }
//         }
//     }
    
//     std::cout << "[CornerClassifier] 共提取 " << seams.size() << " 条焊缝" << std::endl;
//     return seams;
// }



// 判断相机是否可以看到角点（通过平行六面体）
// bool CornerClassifier::isPointInParallelepiped(const Eigen::Vector3f& point,
//                                                 const CornerPoint& corner,
//                                                 const std::vector<FinitePlane>& planes) {
//     // 构建三个方向向量（从角点到三个平面中心）
//     std::vector<Eigen::Vector3f> directions;
//     for (int pid : corner.plane_ids) {
//         const auto& plane = planes[pid];
//         Eigen::Vector3f dir = (plane.center - corner.position);
//         float norm = dir.norm();
//         if (norm < 1e-6f) continue;
//         directions.push_back(dir / norm);
//     }
    
//     if (directions.size() < 3) return false;
    
//     // 将点转换到角点坐标系
//     Eigen::Vector3f local_pt = point - corner.position;
    
//     // 求解线性方程组 local_pt = a*d1 + b*d2 + c*d3
//     Eigen::Matrix3f D;
//     for (int i = 0; i < 3; i++) {
//         D.col(i) = directions[i];
//     }
    
//     // 检查矩阵是否可逆
//     float det = D.determinant();
//     if (std::abs(det) < 1e-4f) return false;
    
//     Eigen::Vector3f coeff = D.inverse() * local_pt;
    
//     // 检查系数是否在合理范围内（考虑一定的容差）
//     float margin = 0.5f;
//     return (coeff.x() >= -margin && coeff.x() <= 1.0f + margin &&
//             coeff.y() >= -margin && coeff.y() <= 1.0f + margin &&
//             coeff.z() >= -margin && coeff.z() <= 1.0f + margin);
// }


bool CornerClassifier::isCameraVisibleToCorner(const CornerPoint& corner,
                                                const std::vector<FinitePlane>& planes,
                                                const Eigen::Vector3f& camera_pos) {
    // ============================================
    // 步骤1：构建局部坐标系
    // ============================================
    // 局部坐标系原点 = 角点位置
    // 三个基向量 = 从角点到三个平面中心的单位向量
    std::vector<Eigen::Vector3f> basis_vectors;
    for (int pid : corner.plane_ids) {
        const auto& plane = planes[pid];
        Eigen::Vector3f vec = plane.center - corner.position;
        float len = vec.norm();
        if (len < 1e-6f) return false;
        basis_vectors.push_back(vec / len);
    }
    
    // ============================================
    // 步骤2：构建变换矩阵 T（世界坐标 → 局部坐标的逆）
    // ============================================
    // T 的列是局部坐标系的基向量在世界坐标系中的表示
    // 即：世界坐标 = T × 局部坐标
    Eigen::Matrix3f T;
    T.col(0) = basis_vectors[0];
    T.col(1) = basis_vectors[1];
    T.col(2) = basis_vectors[2];
    
    // 检查 T 是否可逆（三个向量线性无关）
    float det = T.determinant();
    if (std::abs(det) < 1e-4f) {
        std::cout << "      [警告] 三个平面中心共面，无法构建局部坐标系" << std::endl;
        return false;
    }
    
    // ============================================
    // 步骤3：将相机位置变换到局部坐标系
    // ============================================
    // 局部坐标 = T⁻¹ × (相机位置 - 原点)
    Eigen::Vector3f local_cam = T.inverse() * (camera_pos - corner.position);
    
    // ============================================
    // 步骤4：判断相机是否在三个平面的正向侧
    // ============================================
    // 如果相机在三个平面中心的方向上，三个分量都为正
    float threshold = 0.01f;
    bool visible = (local_cam.x() > threshold && 
                    local_cam.y() > threshold && 
                    local_cam.z() > threshold);
    
    if (visible) {
        std::cout << "      [可见] 局部坐标: ("
                  << local_cam.x() << ", " << local_cam.y() << ", " << local_cam.z() << ")" << std::endl;
    }
    
    return visible;
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
    
    // ✅ 改进：为每个平面选择合适的参考点（不一定是中心点）
    std::vector<Eigen::Vector3f> plane_points;
    
    // 先获取三个平面的大致方向（用于采样）
    std::vector<Eigen::Vector3f> dirs;
    for (int pid : corner.plane_ids) {
        const auto& plane = planes[pid];
        Eigen::Vector3f dir = (plane.center - corner.position).normalized();
        dirs.push_back(dir);
    }
    
    // 为每个平面选择最佳参考点
    for (size_t idx = 0; idx < corner.plane_ids.size(); idx++) {
        int pid = corner.plane_ids[idx];
        const auto& plane = planes[pid];
        
        // 获取另外两个方向
        Eigen::Vector3f other1 = dirs[(idx + 1) % 3];
        Eigen::Vector3f other2 = dirs[(idx + 2) % 3];
        
        Eigen::Vector3f ref_point = getValidPointInPlane(plane, corner.position, other1, other2);
        plane_points.push_back(ref_point);
    }
    
    // 计算三个参考点的向量和
    Eigen::Vector3f sum_points(0, 0, 0);
    for (size_t i = 0; i < plane_points.size(); i++) {
        sum_points += (plane_points[i] - corner.position);
    }
    sum_points.normalize();
    
    float total_score = 0.0f;
    int valid_cameras = 0;
    
    for (const auto& cam_pos : camera_poses) {
        // 检查相机是否可以看到这个角点
        if (!isCameraVisibleToCorner(corner, planes, cam_pos)) {
            continue;
        }
        
        valid_cameras++;
        
        // 计算从角点到相机的向量
        Eigen::Vector3f to_camera = (cam_pos - corner.position);
        if (to_camera.norm() < 1e-6f) continue;
        to_camera.normalize();
        
        // 计算夹角
        float dot = sum_points.dot(to_camera);
        dot = std::max(-1.0f, std::min(1.0f, dot));
        float angle_deg = std::acos(dot) * 180.0f / M_PI;
        
        // 贡献分数
        float similarity = std::cos(angle_deg * M_PI / 180.0f);
        total_score += similarity;
    }
    
    if (valid_cameras == 0) return 0.0f;
    
    return total_score / valid_cameras;
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
