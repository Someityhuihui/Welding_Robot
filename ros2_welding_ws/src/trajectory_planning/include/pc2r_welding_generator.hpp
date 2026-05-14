// pc2r_welding_generator.hpp
#ifndef PC2R_WELDING_GENERATOR_HPP
#define PC2R_WELDING_GENERATOR_HPP

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <functional>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>

#include "adaptive_pointcloud_processor.hpp"

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr;

// 对应点对
struct CorrespondingPointPair {
    int source_idx;
    int target_idx;
    float distance;
};

// 八叉树节点
struct OctreeNode {
    Eigen::Vector3f min_pt;
    Eigen::Vector3f max_pt;
    std::vector<int> point_indices;
    OctreeNode* children[8];
    bool is_leaf;
    
    OctreeNode() : is_leaf(true) {
        for (int i = 0; i < 8; i++) children[i] = nullptr;
    }
};

// 焊接路径点
struct WeldingPathPoint {
    Eigen::Vector3f position;           // 位置
    Eigen::Vector3f torch_direction;    // 焊枪方向 o
    Eigen::Vector3f advance_direction;  // 前进方向 a
    Eigen::Vector3f normal;             // 法向量 n = o × a
    float dihedral_angle;               // 二面角
};

// IBNN (Iterative Bilateral Nearest Neighbor)
class IBNN {
public:
    IBNN();
    ~IBNN();
    
    std::vector<CorrespondingPointPair> findCorrespondingPoints(
        CloudPtr source_cloud,
        CloudPtr target_cloud,
        float epsilon);
    
    float estimateOverlapRegion(CloudPtr source_cloud, CloudPtr target_cloud);
    float adaptiveEpsilon(CloudPtr source_cloud, CloudPtr target_cloud);
    float computePointCloudDensity(CloudPtr cloud);
    
private:
    OctreeNode* root_;
    CloudPtr target_cloud_;
    
    OctreeNode* buildOctree(CloudPtr cloud, const Eigen::Vector3f& min_pt,
                            const Eigen::Vector3f& max_pt, int max_depth);
    void buildOctreeWithPoints(OctreeNode* node, CloudPtr cloud, int max_depth);
    void deleteOctree(OctreeNode* node);
    std::vector<int> voxelSearch(OctreeNode* node, const Eigen::Vector3f& point, float radius);
};

// LM-ICP (Levenberg-Marquardt ICP)
class LMICP {
public:
    LMICP();
    ~LMICP();
    
    bool align(CloudPtr source, CloudPtr target,
               Eigen::Matrix4f& transformation,
               int max_iterations = 50, float tolerance = 1e-6);
    
    bool alignWithInitial(CloudPtr source, CloudPtr target,
                          const Eigen::Matrix4f& initial_transform,
                          Eigen::Matrix4f& final_transform,
                          int max_iterations = 50, float tolerance = 1e-6);
    
private:
    Eigen::Matrix4f current_transform_;
    IBNN ibnn_;
    
    Eigen::MatrixXf computeJacobian(CloudPtr source, CloudPtr target,
                                    const std::vector<CorrespondingPointPair>& pairs,
                                    const Eigen::Matrix4f& transform);
    
    Eigen::VectorXf computeResiduals(CloudPtr source, CloudPtr target,
                                     const std::vector<CorrespondingPointPair>& pairs,
                                     const Eigen::Matrix4f& transform);
};

// 点云拼接器
class PointCloudStitcher {
public:
    PointCloudStitcher();
    ~PointCloudStitcher();
    
    struct CloudData {
        CloudPtr cloud;
        std::string name;
    };
    
    void addPointCloud(CloudPtr cloud, const std::string& name);
    void setStitchingOrder(const std::vector<int>& order);
    CloudPtr stitch();
    
private:
    std::vector<CloudData> clouds_;
    std::vector<int> stitching_order_;
    LMICP lmicp_;
    
    bool hasSufficientOverlap(CloudPtr cloud1, CloudPtr cloud2, float overlap_threshold);
    CloudPtr fuseClouds(CloudPtr cloud1, CloudPtr cloud2, const Eigen::Matrix4f& transform);
};

// 边缘强度检测器 (EI Response)
class EdgeIntensityDetector {
public:
    EdgeIntensityDetector();
    ~EdgeIntensityDetector();
    
    void setRadius(float r) { radius_ = r; }
    void setMinNeighbors(int n) { min_neighbors_ = n; }
    void setEpsilon(float e) { epsilon_ = e; }
    void setSigma(float s) { sigma_ = s; }
    void setIntensityThreshold(float t) { intensity_threshold_ = t; }
    
    float computeIntensity(CloudPtr cloud, int idx, float r = -1);
    Eigen::Vector3f computeGradient(CloudPtr cloud, int idx, float r = -1);
    float computeResponse(CloudPtr cloud, int idx);
    CloudPtr extractEdgePoints(CloudPtr cloud, float threshold = 0.02f);

    void setTargetEdgePoints(int target) { target_edge_points_ = target; }
    void setAutoAdjustPoints(bool enable) { auto_adjust_points_ = enable; }
    
private:
    float radius_;
    int min_neighbors_;
    float epsilon_;
    float sigma_;
    float intensity_threshold_;
    
    Eigen::Matrix3f buildHessianMatrix(const Eigen::Vector3f& grad);
    float computeEigenvalueRatio(const Eigen::Matrix3f& H);

    int target_edge_points_ = 500;  // 目标边缘点数
    bool auto_adjust_points_ = true; // 是否自动调整
};

// B样条拟合器
class BSplineFitter {
public:
    BSplineFitter();
    ~BSplineFitter();
    
    void setDegree(int d) { degree_ = d; }
    void setSamplingRate(float r) { sampling_rate_ = r; }
    
    CloudPtr fitCurve(CloudPtr control_points);
    CloudPtr sampleCurve(float step_size);
    
private:
    int degree_;
    float sampling_rate_;
    CloudPtr control_points_;
    CloudPtr fitted_curve_;
    std::vector<float> knot_vector_;
    
    float basisFunction(int i, int k, float u, const std::vector<float>& knots);
    Eigen::Vector3f computeCurvePoint(float u, CloudPtr control_points,
                                      const std::vector<float>& knots);
};

// 焊枪姿态估计器
class TorchPoseEstimator {
public:
    TorchPoseEstimator();
    ~TorchPoseEstimator();
    
    void setMaxTiltAngle(float angle) { max_tilt_angle_ = angle; }
    void setTiltPeriod(float period) { tilt_period_ = period; }
    void setCameraPosition(const Eigen::Vector3f& pos) { camera_position_ = pos; }
    
    std::vector<WeldingPathPoint> computeTorchPoses(CloudPtr path_points, CloudPtr wsr_cloud);
    
private:
    float max_tilt_angle_;
    float tilt_period_;
    float tilt_offset_;
    Eigen::Vector3f camera_position_;
    
    Eigen::Vector3f estimateNormalPCA(CloudPtr cloud, const std::vector<int>& indices);
    Eigen::Vector3f estimateNormal(CloudPtr cloud, int idx, float radius);
    float computeDihedralAngle(const Eigen::Vector3f& normal1, const Eigen::Vector3f& normal2);
    Eigen::Vector3f computeAdvanceDirection(CloudPtr path_points, int idx);
    void applyTilt(WeldingPathPoint& point, float progress, bool is_uphill);
    float computeLocalCurvature(CloudPtr cloud, const std::vector<int>& indices);
};

// PC2R焊接路径生成器主类
class PC2RWeldingGenerator {
public:
    PC2RWeldingGenerator();
    ~PC2RWeldingGenerator();
    
    // 初始化
    void initialize(float point_cloud_stitch_epsilon = 0.01f,
                    float edge_detection_radius = 0.012f,
                    float edge_threshold = 2.2f,
                    int bspline_degree = 3);
    
    // 设置自适应处理
    void setAdaptiveConfig(const AdaptiveConfig& config);
    void enableAdaptiveProcessing(bool enable) { use_adaptive_ = enable; }
    
    // 参数设置
    void setEdgeDetectionParams(float radius, float epsilon, float threshold, int min_neighbors);
    void setWSRParams(float cylinder_radius);
    void setSortingParams(float neighbor_radius);
    
    // 主要接口
    std::vector<WeldingPathPoint> generateWeldingPath(
        const std::vector<CloudPtr>& multi_view_clouds,
        const std::vector<Eigen::Vector3f>& camera_poses,
        rclcpp::Logger logger,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher = nullptr);
    
    std::vector<WeldingPathPoint> generateWeldingPathFromSingleCloud(
        CloudPtr cloud,
        const Eigen::Vector3f& camera_pose,
        rclcpp::Logger logger);
    
    void saveIntermediateResults(const std::string& output_dir);
    
    // 获取中间结果
    CloudPtr getStitchedCloud() const { return stitched_cloud_; }
    CloudPtr getWSRCloud() const { return wsr_cloud_; }
    CloudPtr getWeldSeamPoints() const { return weld_seam_points_; }
    CloudPtr getPathPoints() const { return path_points_; }
    CloudPtr getAlignedCloud() const { return aligned_cloud_; }

    // 新增 setter 方法
    void setAlignedCloud(CloudPtr cloud) { aligned_cloud_ = cloud; }
    
private:
    // 成员变量
    CloudPtr stitched_cloud_;
    CloudPtr aligned_cloud_;
    CloudPtr wsr_cloud_;
    CloudPtr weld_seam_points_;
    CloudPtr sorted_edge_points_;
    CloudPtr path_points_;
    
    
    std::string output_dir_;
    AdaptivePointCloudProcessor adaptive_processor_;
    AdaptiveConfig adaptive_config_;
    bool use_adaptive_ = true;
    
    // 参数
    float stitch_epsilon_;
    float edge_radius_;
    float edge_threshold_;
    int bspline_degree_;
    float wsr_cylinder_radius_ = 0.16f;      // WSR圆柱滤波半径
    float sorting_neighbor_radius_ = 0.035f; // MST排序邻域半径
    
    // 子模块
    EdgeIntensityDetector edge_detector_;
    BSplineFitter bspline_fitter_;
    TorchPoseEstimator pose_estimator_;
    
    // 私有方法
    CloudPtr step1_StitchPointClouds(const std::vector<CloudPtr>& clouds,
                                     const std::vector<Eigen::Vector3f>& camera_poses);
    
    CloudPtr step1_StitchPointCloudsAdaptive(
    const std::vector<CloudPtr>& clouds,
    const std::vector<Eigen::Vector3f>& camera_poses,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher = nullptr);
    
    CloudPtr step2_ExtractWSR(CloudPtr full_cloud);
    CloudPtr step3_ExtractWeldSeamFeatures(CloudPtr wsr_cloud);
    CloudPtr step4_SortPathPoints(CloudPtr edge_points);
    CloudPtr step5_FitBSplineCurve(CloudPtr sorted_points);
    std::vector<WeldingPathPoint> step6_EstimateTorchPoses(CloudPtr path_points, CloudPtr wsr_cloud);
    
    // 辅助函数
    CloudPtr passThroughFilter(CloudPtr cloud, const Eigen::Vector3f& center, float radius);
    CloudPtr cylinderFilter(CloudPtr cloud, float radius);
    CloudPtr minimumSpanningTreeSort(CloudPtr points, float neighbor_radius);
    CloudPtr greedyNearestNeighborSort(CloudPtr points);
};

// 辅助函数
geometry_msgs::msg::Pose weldingPathPointToPose(const WeldingPathPoint& point);

#endif // PC2R_WELDING_GENERATOR_HPP
