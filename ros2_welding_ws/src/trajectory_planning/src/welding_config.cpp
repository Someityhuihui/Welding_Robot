// ============================================
// welding_config.cpp - 配置实现
// ============================================
#include "welding_config.hpp"
#include <iostream>

// 全局配置实例
WeldingConfig g_config;

void loadConfig() {
    // 如果需要从文件加载，可以在这里实现
    // 目前使用默认值，已在结构体中定义
    std::cout << "Config loaded with default values" << std::endl;
}

void printConfig() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "WELDING TRAJECTORY PLANNING CONFIGURATION" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n[Data Path]" << std::endl;
    std::cout << "  dataset_folder: " << g_config.data_path.dataset_folder << std::endl;
    std::cout << "  output_folder: " << g_config.data_path.output_folder << std::endl;
    
    std::cout << "\n[Point Cloud Acquisition]" << std::endl;
    std::cout << "  max_frames: " << g_config.acquisition.max_frames << std::endl;
    std::cout << "  min_frames: " << g_config.acquisition.min_frames << std::endl;
    
    std::cout << "\n[Adaptive Downsample]" << std::endl;
    std::cout << "  target_points: " << g_config.downsample.target_points_min 
              << " - " << g_config.downsample.target_points_max << std::endl;
    std::cout << "  voxel_size: " << g_config.downsample.voxel_size_min 
              << " - " << g_config.downsample.voxel_size_max << " m" << std::endl;
    
    std::cout << "\n[Point Cloud Stitching (IBNN-LM-ICP)]" << std::endl;
    std::cout << "  icp_max_iterations: " << g_config.stitching.icp_max_iterations << std::endl;
    std::cout << "  icp_fitness_threshold: " << g_config.stitching.icp_fitness_threshold << std::endl;
    std::cout << "  min_overlap_ratio: " << g_config.stitching.min_overlap_ratio << std::endl;
    
    std::cout << "\n[Edge Detection (EI Response)]" << std::endl;
    std::cout << "  search_radius: " << g_config.edge.search_radius << " m" << std::endl;
    std::cout << "  epsilon: " << g_config.edge.epsilon << std::endl;
    std::cout << "  intensity_threshold: " << g_config.edge.intensity_threshold << std::endl;
    
    std::cout << "\n[Path Sorting (MST)]" << std::endl;
    std::cout << "  neighbor_radius: " << g_config.sorting.neighbor_radius << " m" << std::endl;
    
    std::cout << "\n[BSpline Fitting]" << std::endl;
    std::cout << "  degree: " << g_config.bspline.degree << std::endl;
    std::cout << "  sampling_rate: " << g_config.bspline.sampling_rate << " m" << std::endl;
    
    std::cout << "\n[Torch Pose Estimation]" << std::endl;
    std::cout << "  max_tilt_angle: " << g_config.torch.max_tilt_angle << " deg" << std::endl;
    std::cout << "  pca_neighbors: " << g_config.torch.pca_neighbors << std::endl;
    
    std::cout << "\n========================================" << std::endl;
}
