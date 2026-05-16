根据您的项目流程，我来生成一个完整的 README.md 文档。

```markdown
# 焊缝自动提取系统 (Plane Weld Vision Extractor)

## 项目简介

基于3D点云数据的焊缝自动识别与提取系统，用于工业机器人焊接路径规划。系统通过多视角点云配准、平面分割、角点检测和焊缝生成，自动识别焊接结构件的焊缝位置和姿态。

## 系统流程

```
原始多帧点云 → 点云配准 → 平面提取 → 角点检测 → 焊缝生成 → 可视化
```

### 第一步：编译

```bash
cd Plane_weld_vision_extractor/build
cmake ..
make
```

编译生成可执行文件 `weld_extractor`。

### 第二步：运行焊缝提取

```bash
./weld_extractor <数据集路径> <输出路径> [选项]
```

**示例：**
```bash
./weld_extractor ../1_models_with_lidar_frame_simulation/seam_fillet_lap_scene_normalized_irregular_dataset_lidar_sim_Z_36_frame/ ../seam_fillet_lap_scene_normalized_irregular_output --method ransac --boundary-margin 5
```

### 第三步：可视化结果

```bash
python3 ../scripts/visualize_complete.py ../seam_fillet_lap_scene_normalized_irregular_output
```

## 算法流程详解

### 步骤1：点云配准 (PointCloudRegistrar)

**功能**：将多帧点云合并到全局坐标系

**流程**：
- 加载所有 `{i}_frame.pcd` 点云文件
- 读取对应的 `{i}_frame.txt` 相机位姿文件（可选）
- 将所有点云转换到全局坐标系并合并
- 输出全局点云 `global_cloud.pcd`

### 步骤2：平面提取 (PlaneExtractor)

**功能**：从点云中提取有限平面，支持两种方法

#### 方法A：RANSAC + 连通域分割

```
原始点云 → 降采样 → 去噪 → RANSAC平面拟合 → 提取内点 → 剩余点云继续
                    ↓
              重复直到剩余点数不足
                    ↓
              连通分量分割
                    ↓
              分裂为独立平面
```

#### 方法B：区域生长法

```
原始点云 → 降采样 → 去噪 → 法向量计算 → 区域生长分割 → 3σ补全 → 连通分量分割
```

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | region | 平面提取方法 (ransac/region) |
| `--min-plane-points` | 500 | 最小平面点数 |
| `--voxel-size` | 0.003 | 降采样体素大小 |

### 步骤3：角点检测 (CornerClassifier)

**功能**：检测三平面交点，区分内凹角（焊缝）和外凸角（边角）

#### 3.1 三平面求交

```cpp
// 求解三个平面方程的交点
A * p = b
其中 A = [n1, n2, n3]^T, b = [-d1, -d2, -d3]^T
```

#### 3.2 内凹角判定（焊缝向量法）

以角点为原点，三条焊缝的方向为基向量构建局部坐标系：

```
v1 = (焊缝1终点 - 角点).normalized()
v2 = (焊缝2终点 - 角点).normalized()
v3 = (焊缝3终点 - 角点).normalized()
```

相机位置在该局部坐标系中的坐标：

```
local_cam = T^{-1} * (camera_pos - corner_point)
```

**判定条件**：相机局部坐标的三个分量都 > 0.01

#### 3.3 边界过滤

过滤距离模型边界太近的角点（这些通常是外凸的边角点）：

```bash
--boundary-margin 5    # 过滤距离边界5mm内的角点
--no-boundary-filter   # 不过滤边界角点
```

### 步骤4：焊缝生成 (WeldSeam)

**功能**：从内凹角生成三条焊缝路径

**流程**：
1. 计算两平面交线
2. 确定交线在有限平面内的有效范围
3. 沿交线采样路径点
4. 计算焊枪姿态（法向量 + 四元数）

## 输出文件说明

| 文件 | 格式 | 内容 |
|------|------|------|
| `global_cloud.pcd` | PCD | 合并后的全局点云 |
| `camera_poses.txt` | TXT | 相机位姿（x,y,z每行） |
| `colored_planes.pcd` | PCD | 彩色平面点云（每个平面不同颜色） |
| `planes/plane_N.pcd` | PCD | 每个平面的独立点云 |
| `corners_info.txt` | CSV | 角点信息（位置、内凹/外凸、置信度） |
| `weld_seams.csv` | CSV | 焊缝信息（位置、法向量、姿态四元数） |

### weld_seams.csv 格式

```csv
weld_id,x,y,z,nx,ny,nz,ox,oy,oz,ow
```

| 字段 | 说明 |
|------|------|
| x,y,z | 焊缝路径点坐标（米） |
| nx,ny,nz | 该点的法向量 |
| ox,oy,oz,ow | 焊枪姿态四元数 |

### corners_info.txt 格式

```csv
id,x,y,z,is_concave,confidence,plane0,plane1,plane2
```

| 字段 | 说明 |
|------|------|
| is_concave | 1=内凹角(焊缝), 0=外凸角(边角) |
| confidence | 置信度 (0-1) |
| plane0-2 | 相交的三个平面ID |

## 命令行选项完整说明

```bash
./weld_extractor <数据集文件夹> <输出文件夹> [选项]

选项:
  --no-poses                   没有位姿文件，使用ICP配准
  --mode corner                边角模式：只提取三平面相交的角焊缝（默认）
  --mode long                  长条模式：只提取两平面相交的长焊缝
  --mode both                  混合模式：提取所有焊缝
  --method ransac              使用RANSAC方法提取平面
  --method region              使用区域生长方法提取平面（默认）
  --min-length <mm>            最小焊缝长度（默认50mm）
  --no-boundary-filter         不过滤边界角点（保留所有角点）
  --boundary-margin <mm>       边界距离阈值（默认10mm）
  --min-plane-points <num>     最小平面点数（默认500）
  --voxel-size <m>             降采样体素大小（默认0.003m）
```

## 使用示例

### 基础用法

```bash
# 默认：区域生长 + 边角模式 + 过滤边界
./weld_extractor ./data ./output

# 混合模式（提取所有焊缝）
./weld_extractor ./data ./output --mode both

# RANSAC方法 + 混合模式
./weld_extractor ./data ./output --method ransac --mode both

# 严格边界过滤（5mm）+ 长条模式
./weld_extractor ./data ./output --mode long --boundary-margin 5

# 不过滤边界 + 混合模式
./weld_extractor ./data ./output --no-boundary-filter --mode both

# 完整示例（您的使用场景）
./weld_extractor ../1_models_with_lidar_frame_simulation/seam_fillet_lap_scene_normalized_irregular_dataset_lidar_sim_Z_36_frame/ ../seam_fillet_lap_scene_normalized_irregular_output --method ransac --boundary-margin 5
```

### 可视化结果

```bash
python3 ../scripts/visualize_complete.py ../seam_fillet_lap_scene_normalized_irregular_output
```

## 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 降采样体素 | 3mm | 平衡精度和速度 |
| RANSAC迭代 | 2000 | 提高平面拟合稳定性 |
| 平面距离阈值 | 10mm | 放宽检测范围 |
| 最小平面点数 | 500 | 过滤小平面噪声 |
| 连通域聚类容差 | 20mm | 分离不连通区域 |
| 焊缝采样步长 | 5mm | 路径点密度 |

## 目录结构

```
Plane_weld_vision_extractor/
├── include/
│   ├── corner_classifier.hpp
│   ├── plane_extractor.hpp
│   ├── pointcloud_registrar.hpp
│   └── weld_extractor.hpp
├── src/
│   ├── corner_classifier.cpp
│   ├── plane_extractor.cpp
│   ├── pointcloud_registrar.cpp
│   └── weld_extractor.cpp
├── scripts/
│   └── visualize_complete.py
├── build/
└── CMakeLists.txt
```

## 依赖库

- **PCL 1.8+**：点云处理（ICP、RANSAC、区域生长）
- **Eigen3**：线性代数计算
- **OpenMP**：并行加速（可选）

安装依赖（Ubuntu）：
```bash
sudo apt-get install libpcl-dev libeigen3-dev
```

## 常见问题

### Q1: 平面提取过多怎么办？
增大 `--min-plane-points` 或使用区域生长法时增大平滑度阈值。

### Q2: 角点识别不准确？
调整 `--boundary-margin` 参数，或使用 `--no-boundary-filter` 查看所有角点。

### Q3: 焊缝长度计算异常？
检查平面边界计算是否正确，确保有限平面分割准确。

### Q4: 编译失败？
确保已安装所有依赖，清理build目录后重新编译：
```bash
cd build && rm -rf * && cmake .. && make -j4
```

## 版本历史

- **v1.0**：基础RANSAC平面提取 + 角点分类
- **v1.1**：添加区域生长法和连通域分割
- **v1.2**：添加3σ补全和焊缝向量验证
- **v1.3**：添加边界过滤和命令行选项

## 许可证

本项目仅供研究和学习使用。

## 联系方式

如有问题，请提交Issue或联系项目维护者。
```