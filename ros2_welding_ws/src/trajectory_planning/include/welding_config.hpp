// welding_config.hpp
#ifndef WELDING_CONFIG_HPP
#define WELDING_CONFIG_HPP

#include <string>

struct DataPathConfig {
    std::string dataset_folder = "/home/someityhuihui/ros2_welding_test_data/seam_3_pipe_normalized_irregular_dataset_5_frame/";
    std::string output_folder = "/home/someityhuihui/ros2_welding_test_data/pc2r_results/";
};

struct PointCloudAcquisition {
    int max_frames = 5;
    int min_frames = 3;
    bool enable_adaptive_stitching = true;
};

struct AdaptiveDownsample {
    size_t target_points_min = 30000;
    size_t target_points_max = 100000;
    float voxel_size_min = 0.001f;
    float voxel_size_max = 0.008f;
    int recursive_limit = 3;
};

struct PointCloudStitching {
    // IBNN参数
    float ibnn_epsilon = 0.008f;
    float min_overlap_ratio = 0.25f;
    
    // ICP参数
    int icp_max_iterations = 50;
    float icp_fitness_threshold = 0.0001f;
    float max_correspondence_distance = 0.05f;
    int ransac_iterations = 200;
    
    // 关键帧检测
    float bbox_change_threshold = 0.15f;
};

struct EdgeDetection {
    float search_radius = 0.050f;
    float epsilon = 0.9f;
    float intensity_threshold = 0.0005f;
    int min_neighbors = 3;
    float sigma = 1.0f;
};

struct WSRExtraction {
    float cylinder_radius = 0.18f;
    std::string method = "cylinder";
};

struct PathSorting {
    float neighbor_radius = 0.035f;    // 3.5cm (增大)
    int min_cluster_size = 10;
};

struct BSplineConfig {
    int degree = 3;
    float sampling_rate = 0.003f;      // 3mm
};

struct TorchPoseConfig {
    float max_tilt_angle = 15.0f;      // 度
    int pca_neighbors = 30;
    float pca_radius = 0.015f;
};

struct PerformanceConfig {
    bool enable_adaptive_processing = true;
    bool enable_keyframe_strategy = true;
    bool enable_parallel = false;
};

struct VisualizationConfig {
    std::string fixed_frame = "base_link";
    float marker_arrow_length = 0.05f;
    float marker_sphere_size = 0.015f;
};

struct WeldingConfig {
    DataPathConfig data_path;
    PointCloudAcquisition acquisition;
    AdaptiveDownsample downsample;
    PointCloudStitching stitching;
    EdgeDetection edge;
    WSRExtraction wsr;
    PathSorting sorting;
    BSplineConfig bspline;
    TorchPoseConfig torch;
    PerformanceConfig perf;
    VisualizationConfig viz;
};

extern WeldingConfig g_config;

void loadConfig();
void printConfig();

#endif // WELDING_CONFIG_HPP
