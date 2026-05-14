#!/usr/bin/env python3
"""
从规则点云生成不规则点云数据集（含噪声 + 视角密度）
用法: python3 generate_irregular_dataset.py <输入点云.pcd> [选项]

选项:
  --frames <N>          生成帧数 (默认: 5)
  --output <dir>        输出目录 (默认: ./irregular_dataset)
  --noise <sigma>       高斯噪声标准差 (默认: 0.0005)
  --camera <x,y,z>      相机位置 (默认: 0.5,0,0.5)
  --density <factor>    密度变化因子 0.3~0.9 (默认: 0.7)

示例:
  # 基本用法
  python3 generate_irregular_dataset.py pointcloud.pcd
  
  # 自定义参数
  python3 generate_irregular_dataset.py pointcloud.pcd --frames 5 --output ./dataset --noise 0.001 --camera 0.3,0.2,0.4 --density 0.8
"""

import sys
import os
import numpy as np
import struct
import random

# ============================================
# 点云读取和写入
# ============================================

def read_pcd_file(filename):
    """自动检测并读取 PCD 文件（支持 ASCII 和二进制格式）"""
    
    with open(filename, 'rb') as f:
        content = f.read()
    
    try:
        header_text = content.decode('utf-8')
    except:
        header_text = content.decode('latin-1')
    
    lines = header_text.split('\n')
    fields = []
    sizes = []
    data_type = None
    data_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            sizes = [int(x) for x in line.split()[1:]]
        elif line.startswith('DATA'):
            data_type = line.split()[1]
            data_start = header_text.find('\n', i) + 1
            break
    
    data_bytes = content[data_start:]
    points = []
    
    if data_type == 'ascii':
        data_str = data_bytes.decode('utf-8', errors='ignore')
        for line in data_str.strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    except:
                        continue
    
    elif data_type == 'binary':
        point_step = sum(sizes)
        for i in range(0, len(data_bytes), point_step):
            if i + point_step <= len(data_bytes):
                point_data = data_bytes[i:i+point_step]
                x = struct.unpack('f', point_data[0:4])[0]
                y = struct.unpack('f', point_data[4:8])[0]
                z = struct.unpack('f', point_data[8:12])[0]
                points.append([x, y, z])
    
    else:
        print(f"错误: 不支持的数据格式 {data_type}")
        return np.array([])
    
    print(f"读取 {len(points)} 个点")
    return np.array(points, dtype=np.float32)

def write_pcd_ascii(filename, points):
    """写入 ASCII 格式的 PCD 文件"""
    with open(filename, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

# ============================================
# 不规则化处理
# ============================================

def add_gaussian_noise(points, noise_level=0.0005):
    """步骤1: 添加高斯噪声（模拟扫描误差）"""
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    
    print(f"\n[步骤1] 添加高斯噪声 (sigma={noise_level})")
    print(f"  原始点数: {len(points)}")
    
    return noisy_points

def simulate_camera_view(points, camera_pos=(0.5, 0, 0.5), density_factor=0.7):
    """
    步骤2: 模拟相机视角（离相机近的点保留率高）
    camera_pos: 相机位置 (x, y, z)
    density_factor: 密度变化因子，越大远处点越少
    """
    # 计算每个点到相机的距离
    distances = []
    for p in points:
        dx = p[0] - camera_pos[0]
        dy = p[1] - camera_pos[1]
        dz = p[2] - camera_pos[2]
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        distances.append(d)
    
    max_dist = max(distances)
    min_dist = min(distances)
    dist_range = max_dist - min_dist
    
    # 计算每个点的保留概率
    probabilities = []
    for d in distances:
        if dist_range > 0:
            dist_norm = (d - min_dist) / dist_range
        else:
            dist_norm = 0
        # 保留概率 = 1 - density_factor * dist_norm
        prob = 1.0 - density_factor * dist_norm
        prob = max(0.1, min(0.95, prob))
        probabilities.append(prob)
    
    # 按概率采样
    sampled_points = []
    for i, p in enumerate(points):
        if random.random() < probabilities[i]:
            sampled_points.append(p)
    
    print(f"\n[步骤2] 模拟相机视角 (相机位置: {camera_pos})")
    print(f"  最近距离: {min_dist:.4f}m, 最远距离: {max_dist:.4f}m")
    print(f"  采样前点数: {len(points)}")
    print(f"  采样后点数: {len(sampled_points)}")
    print(f"  保留率: {len(sampled_points)/len(points)*100:.1f}%")
    
    return np.array(sampled_points, dtype=np.float32)

def make_irregular(points, noise_level, camera_pos, density_factor):
    """组合步骤1和2"""
    print("\n" + "=" * 50)
    print("开始处理点云...")
    print("=" * 50)
    
    # 步骤1: 添加噪声
    points = add_gaussian_noise(points, noise_level)
    
    # 步骤2: 相机视角密度
    points = simulate_camera_view(points, camera_pos, density_factor)
    
    return points

# ============================================
# 多帧数据集生成
# ============================================

def generate_frames(points, num_frames, output_dir):
    """生成多帧数据集"""
    
    # 计算范围
    center = np.mean(points, axis=0)
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    z_range = np.max(points[:, 2]) - np.min(points[:, 2])
    
    print(f"\n点云中心: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"范围: x={x_range:.4f}, y={y_range:.4f}, z={z_range:.4f}")
    print(f"\n生成 {num_frames} 帧数据...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成每一帧
    for frame in range(1, num_frames + 1):
        angle = (frame - 1) * (2 * np.pi / num_frames)
        
        # 旋转矩阵（绕 Y 轴轻微旋转）
        cos_a = np.cos(angle * 0.1)
        sin_a = np.sin(angle * 0.1)
        rotation = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # 变换点云
        transformed = points @ rotation.T
        
        # 添加轻微平移
        translation = np.array([
            np.sin(angle) * 0.05,
            0,
            np.cos(angle) * 0.05
        ])
        transformed = transformed + translation
        
        # 保存 PCD
        pcd_file = os.path.join(output_dir, f"{frame}_frame.pcd")
        write_pcd_ascii(pcd_file, transformed)
        
        # 生成相机位置
        camera_radius = max(x_range, y_range, z_range) * 1.5
        camera_x = center[0] + camera_radius * np.cos(angle)
        camera_z = center[2] + camera_radius * np.sin(angle)
        camera_y = center[1] + 0.2
        
        # 保存 TXT
        txt_file = os.path.join(output_dir, f"{frame}_frame.txt")
        with open(txt_file, 'w') as f:
            f.write(f"{camera_x:.6f}\t{camera_y:.6f}\t{camera_z:.6f}\n")
        
        print(f"  生成第 {frame} 帧")
    
    return output_dir

# ============================================
# 主函数
# ============================================

def main():
    # 参数解析
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 默认参数
    num_frames = 5
    output_dir = "./irregular_dataset"
    noise_level = 0.0005
    camera_pos = (0.5, 0, 0.5)
    density_factor = 0.7
    
    # 解析可选参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--frames' and i+1 < len(sys.argv):
            num_frames = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--output' and i+1 < len(sys.argv):
            output_dir = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--noise' and i+1 < len(sys.argv):
            noise_level = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--camera' and i+1 < len(sys.argv):
            parts = sys.argv[i+1].split(',')
            if len(parts) == 3:
                camera_pos = (float(parts[0]), float(parts[1]), float(parts[2]))
            i += 2
        elif sys.argv[i] == '--density' and i+1 < len(sys.argv):
            density_factor = float(sys.argv[i+1])
            i += 2
        else:
            i += 1
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("不规则点云数据集生成器")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"帧数: {num_frames}")
    print(f"噪声级别: {noise_level}")
    print(f"相机位置: {camera_pos}")
    print(f"密度因子: {density_factor}")
    
    # 读取规则点云
    print("\n读取规则点云...")
    points = read_pcd_file(input_file)
    
    if len(points) == 0:
        print("错误: 无法读取点云数据")
        sys.exit(1)
    
    print(f"原始点云范围:")
    print(f"  X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
    print(f"  Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
    print(f"  Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")
    
    # 转换为不规则点云
    irregular_points = make_irregular(points, noise_level, camera_pos, density_factor)
    
    if len(irregular_points) == 0:
        print("错误: 不规则化处理失败")
        sys.exit(1)
    
    # 生成数据集
    output_path = generate_frames(irregular_points, num_frames, output_dir)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"数据集目录: {os.path.abspath(output_path)}")
    print(f"\n在 main.cpp 中设置:")
    print(f'string dataset_folder_path = "{os.path.abspath(output_path)}/";')

if __name__ == "__main__":
    main()
