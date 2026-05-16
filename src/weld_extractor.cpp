// weld_extractor.cpp
#include "weld_extractor.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <cstring>
#include <string>
#include <pcl/io/pcd_io.h> 

WeldExtractor::WeldExtractor() {}

WeldExtractor::~WeldExtractor() {}


bool WeldExtractor::loadFrames(const std::string& folder_path, bool has_pose_files) {
    return registrar_.loadFrames(folder_path, has_pose_files);
}

bool WeldExtractor::registerPointClouds() {
    return registrar_.registerPointClouds();
}

std::vector<WeldSeam> WeldExtractor::extractWeldSeams(CloudPtr cloud, 
                                                        const std::vector<Eigen::Vector3f>& camera_poses) {
    std::cout << "\n========== 步骤2：焊缝提取 ==========" << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    // 设置分类器
    classifier_.setPointCloud(cloud);
    classifier_.setCameraPoses(camera_poses);
    
    // 步骤2.1：提取有限平面
    std::cout << "[步骤2.1] 提取有限平面..." << std::endl;
    
    std::vector<FinitePlane> planes;
    
    if (extraction_method_ == METHOD_REGION_GROWING) {
        std::cout << "[步骤2.1] 使用区域生长法..." << std::endl;
        planes = plane_extractor_.extractPlanesRegionGrowing(cloud);
    } else {
        std::cout << "[步骤2.1] 使用 RANSAC + 连通分量法..." << std::endl;
        planes = plane_extractor_.extractPlanes(cloud);
    }

     // 输出平面统计
    std::cout << "\n[平面统计] 共 " << planes.size() << " 个平面:" << std::endl;
    for (const auto& p : planes) {
        std::cout << "  平面 " << p.id << ": 法向量=(" << p.normal.x() << "," << p.normal.y() << "," << p.normal.z()
                  << "), 中心=(" << p.center.x() << "," << p.center.y() << "," << p.center.z() 
                  << "), 点数=" << p.point_count << std::endl;
    }

    if (planes.empty()) {
        std::cerr << "[错误] 未提取到任何平面" << std::endl;
        return {};
    }
    
    // ===== 关键新增：统一法向量朝外（必须在任何交点计算前执行）=====
    if (!camera_poses.empty()) {
        std::cout << "[步骤2.1b] 统一法向量方向（相机投票）..." << std::endl;
        plane_extractor_.orientNormalsOutward(planes, camera_poses);
    } else {
        std::cout << "[步骤2.1b] 警告：无相机位姿，法向量方向未统一，角点分类可能失效" << std::endl;
    }

    // ===== 新增：可视化每个有限平面（不同颜色）=====
    std::cout << "\n[可视化] 生成平面彩色点云..." << std::endl;
    
    // 创建彩色点云用于可视化
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (const auto& plane : planes) {
        if (!plane.cloud || plane.cloud->points.empty()) continue;
        
        for (const auto& p : plane.cloud->points) {
            pcl::PointXYZRGB rgb_p;
            rgb_p.x = p.x;
            rgb_p.y = p.y;
            rgb_p.z = p.z;
            rgb_p.r = plane.r;
            rgb_p.g = plane.g;
            rgb_p.b = plane.b;
            colored_cloud->push_back(rgb_p);
        }
    }
    
    // 保存彩色点云
    std::string plane_vis_path = output_folder_ + "/colored_planes.pcd";
    pcl::io::savePCDFileBinary(plane_vis_path, *colored_cloud);
    std::cout << "[可视化] 保存平面彩色点云: " << plane_vis_path << std::endl;
    std::cout << "[可视化] 共 " << planes.size() << " 个平面，"
              << colored_cloud->points.size() << " 个彩色点" << std::endl;
    
    // 可选：同时保存每个平面独立的点云
    std::string planes_dir = output_folder_ + "/planes";
    std::filesystem::create_directories(planes_dir);
    for (const auto& plane : planes) {
        if (!plane.cloud || plane.cloud->points.empty()) continue;
        std::string plane_path = planes_dir + "/plane_" + std::to_string(plane.id) + ".pcd";
        pcl::io::savePCDFileBinary(plane_path, *plane.cloud);
        std::cout << "  保存平面 " << plane.id << " 到: " << plane_path << std::endl;
    }
    
    // 步骤2.2：计算所有三平面交点
    std::cout << "[步骤2.2] 计算三平面交点..." << std::endl;
    std::vector<Eigen::Vector3f> intersections;
    std::vector<std::tuple<int, int, int>> intersection_plane_indices;  // ✅ 新增

    for (size_t i = 0; i < planes.size(); i++) {
        for (size_t j = i + 1; j < planes.size(); j++) {
            for (size_t k = j + 1; k < planes.size(); k++) {
                Eigen::Vector3f pt;
                if (plane_extractor_.computeTriplePlaneIntersection(planes[i], planes[j], planes[k], pt)) {
                    intersections.push_back(pt);
                    intersection_plane_indices.push_back({i, j, k});  // ✅ 新增
                    std::cout << "  [交点] 平面(" << i << "," << j << "," << k << ") -> ("
                            << pt.x() << ", " << pt.y() << ", " << pt.z() << ")" << std::endl;
                }
            }
        }
    }
    std::cout << "[步骤2.2] 找到 " << intersections.size() << " 个有效三平面交点" << std::endl;

    // 步骤2.3：角点分类（内凹/外凸）
    std::cout << "[步骤2.3] 角点分类..." << std::endl;
    auto corners = classifier_.classifyCorners(planes, intersections, intersection_plane_indices);  // ✅ 传入三个参数
    
    int concave_count = 0, convex_count = 0;
    for (const auto& c : corners) {
        if (c.is_concave) concave_count++;
        else convex_count++;
    }
    std::cout << "[步骤2.3] 角点分类完成：内凹(焊缝)=" << concave_count 
              << ", 外凸(非焊缝)=" << convex_count << std::endl;

    // ✅ 保存角点信息到输出目录
    std::string corners_path = output_folder_ + "/corners_info.txt";
    classifier_.saveCornersToFile(corners, corners_path);
    
    // 步骤2.4：提取焊缝
    std::cout << "[步骤2.4] 提取焊缝..." << std::endl;
    auto seams = classifier_.extractWeldSeams(planes, corners);
    
    // 统计焊缝类型
    int corner_weld_count = 0, butt_weld_count = 0;
    float total_corner_length = 0.0f, total_butt_length = 0.0f;
    for (const auto& s : seams) {
        if (s.is_corner_weld) {
            corner_weld_count++;
            total_corner_length += s.length;
        } else {
            butt_weld_count++;
            total_butt_length += s.length;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "\n========== 提取完成 ==========" << std::endl;
    std::cout << "有限平面数: " << planes.size() << std::endl;
    // 保存完成后输出统计信息
    std::cout << "\n========== 平面可视化统计 ==========" << std::endl;
    std::cout << "平面数量: " << planes.size() << std::endl;
    for (const auto& plane : planes) {
        std::cout << "  平面 " << plane.id << ": 点数=" << plane.point_count
                << ", 颜色=RGB(" << plane.r << "," << plane.g << "," << plane.b << ")" << std::endl;
    }
    std::cout << "三平面交点: " << intersections.size() << std::endl;
    std::cout << "有效角点:   " << corners.size() << " (内凹=" << concave_count << ", 外凸=" << convex_count << ")" << std::endl;
    std::cout << "焊缝总数:   " << seams.size() << std::endl;
    std::cout << "  └─ 角焊缝: " << corner_weld_count << " 条, 总长 " << total_corner_length * 1000.0f << " mm" << std::endl;
    std::cout << "  └─ 长焊缝: " << butt_weld_count << " 条, 总长 " << total_butt_length * 1000.0f << " mm" << std::endl;
    std::cout << "耗时: " << elapsed << " 秒" << std::endl;
    
    seams_ = seams;
    return seams;
}


bool WeldExtractor::process(const std::string& dataset_folder, 
                             const std::string& output_folder,
                             bool has_pose_files) {
    std::cout << "\n########################################" << std::endl;
    std::cout << "# 焊缝自动提取系统" << std::endl;
    std::cout << "########################################" << std::endl;
    
    output_folder_ = output_folder;
    
    // 创建输出目录
    std::filesystem::create_directories(output_folder);
    
    // 步骤1：点云配准
    std::cout << "\n========== 步骤1：点云配准 ==========" << std::endl;
    if (!loadFrames(dataset_folder, has_pose_files)) {
        std::cerr << "[错误] 加载点云失败" << std::endl;
        return false;
    }
    
    if (!registerPointClouds()) {
        std::cerr << "[错误] 点云配准失败" << std::endl;
        return false;
    }
    
    // 保存中间结果
    saveGlobalCloud(output_folder + "/global_cloud.pcd");
    saveCameraPoses(output_folder + "/camera_poses.txt");
    
    // 步骤2：焊缝提取
    auto seams = extractWeldSeams(registrar_.getGlobalCloud(), registrar_.getCameraPoses());
    
    if (seams.empty()) {
        std::cerr << "[错误] 未提取到焊缝" << std::endl;
        return false;
    }
    
    // 保存结果
    saveWeldSeams(output_folder + "/weld_seams.csv");
    
    std::cout << "\n########################################" << std::endl;
    std::cout << "# 处理完成！" << std::endl;
    std::cout << "# 输出文件:" << std::endl;
    std::cout << "#   " << output_folder << "/global_cloud.pcd" << std::endl;
    std::cout << "#   " << output_folder << "/camera_poses.txt" << std::endl;
    std::cout << "#   " << output_folder << "/weld_seams.csv" << std::endl;
    std::cout << "########################################" << std::endl;
    
    return true;
}

bool WeldExtractor::saveGlobalCloud(const std::string& filename) {
    return registrar_.saveGlobalCloud(filename);
}

bool WeldExtractor::saveWeldSeams(const std::string& filename) {
    return classifier_.exportWeldSeamsToCSV(seams_, filename);
}


bool WeldExtractor::saveCameraPoses(const std::string& filename) {
    return registrar_.saveCameraPoses(filename);
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "用法: weld_extractor <数据集文件夹> <输出文件夹> [选项]" << std::endl;
        std::cout << std::endl;
        
        std::cout << "选项:" << std::endl;
        std::cout << "  --no-poses           没有位姿文件，使用ICP配准" << std::endl;
        std::cout << "  --mode corner        边角模式：只提取三平面相交的三条焊缝" << std::endl;
        std::cout << "  --mode long          长条模式：只提取两平面相交的长焊缝" << std::endl;
        std::cout << "  --mode both          混合模式：提取所有焊缝（默认）" << std::endl;

        std::cout << "  --method ransac      使用RANSAC方法提取平面（默认）" << std::endl;
        std::cout << "  --method region      使用区域生长方法提取平面" << std::endl;
        std::cout << "  --min-length <mm>    最小焊缝长度（单位：mm，默认50）" << std::endl;
        std::cout << std::endl;

        std::cout << "示例:" << std::endl;
        std::cout << "  weld_extractor ./data ./output --mode corner" << std::endl;
        std::cout << "  weld_extractor ./data ./output --mode long --min-length 100" << std::endl;
        std::cout << "  weld_extractor ./data ./output --method region --mode both" << std::endl;
        std::cout << "选项:" << std::endl;

        std::cout << "  --no-boundary-filter       不过滤边界角点（保留所有角点）" << std::endl;
        std::cout << "  --boundary-margin <mm>     边界过滤距离阈值（单位：mm，默认10）" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  weld_extractor ./data ./output                                    # 默认过滤边界（10mm）" << std::endl;
        std::cout << "  weld_extractor ./data ./output --no-boundary-filter               # 不过滤边界" << std::endl;
        std::cout << "  weld_extractor ./data ./output --boundary-margin 5                # 边界阈值5mm" << std::endl;
        std::cout << "  weld_extractor ./data ./output --no-boundary-filter --mode both   # 不过滤边界+混合模式" << std::endl;
        return 1;
    }
    
    std::string dataset_folder = argv[1];
    std::string output_folder = argv[2];
    bool has_pose_files = true;
    ExtractionMode mode = MODE_CORNER_ONLY; // 默认边角模式
    ExtractionMethod extraction_method = METHOD_REGION_GROWING; // ✅ 添加变量声明，默认REGION_GROWING
    bool use_icp = false;
    float min_length_mm = 50.0f;

    bool filter_boundary = true;      // 默认过滤边界
    float boundary_margin_mm = 10.0f; // 默认10mm
    
    // 解析参数
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--no-poses") == 0) {
            has_pose_files = false;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "corner") == 0) {
                mode = MODE_CORNER_ONLY;
                std::cout << "提取模式: 边角模式" << std::endl;
            } else if (strcmp(argv[i], "long") == 0) {
                mode = MODE_LONG_ONLY;
                std::cout << "提取模式: 长条模式" << std::endl;
            } else if (strcmp(argv[i], "both") == 0) {
                mode = MODE_BOTH;
                std::cout << "提取模式: 混合模式" << std::endl;
            }
        } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "ransac") == 0) {
                extraction_method = METHOD_RANSAC;
                std::cout << "平面提取方法: RANSAC" << std::endl;
            } else if (strcmp(argv[i], "region") == 0) {
                extraction_method = METHOD_REGION_GROWING;
                std::cout << "平面提取方法: 区域生长" << std::endl;
            }
        } else if (strcmp(argv[i], "--min-length") == 0 && i + 1 < argc) {
            i++;
            min_length_mm = std::stof(argv[i]);
            std::cout << "最小焊缝长度: " << min_length_mm << " mm" << std::endl;
        }  else if (strcmp(argv[i], "--no-boundary-filter") == 0) {
            filter_boundary = false;
            std::cout << "  [设置] 边界过滤: 禁用" << std::endl;
        } else if (strcmp(argv[i], "--boundary-margin") == 0 && i + 1 < argc) {
            i++;
            boundary_margin_mm = std::stof(argv[i]);
            std::cout << "  [设置] 边界距离阈值: " << boundary_margin_mm << " mm" << std::endl;
        }
    }
    
    WeldExtractor extractor;
    
    // 设置是否使用ICP配准
    extractor.setUseICP(use_icp);
    
    // 设置参数
    extractor.setVoxelSize(0.003f);
    extractor.setPlaneThreshold(0.005f);
    extractor.setMinPlanePoints(500);
    extractor.setPathSpacing(0.005f);
    extractor.setWeldAngle(45.0f);
    extractor.setExtractionMode(mode);
    extractor.setExtractionMethod(extraction_method);  // ✅ 设置提取方法
    extractor.setMinWeldLength(min_length_mm / 1000.0f);  // 转换为米
    // 设置边界过滤参数
    extractor.setFilterBoundaryCorners(filter_boundary);
    extractor.setBoundaryMargin(boundary_margin_mm);
    
    if (extractor.process(dataset_folder, output_folder, has_pose_files)) {
        std::cout << "\n✅ 处理成功！" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ 处理失败！" << std::endl;
        return 1;
    }
}
