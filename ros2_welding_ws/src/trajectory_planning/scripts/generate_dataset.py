#!/usr/bin/env python3
"""
从单个点云文件生成模拟数据集（支持 ASCII 和二进制 PCD 格式）
用法: python3 generate_dataset.py <输入点云.pcd> [帧数] [输出目录]

'''
实例:
# 点云先做归一化
python3 normalize_pcd.py "/mnt/c/Users/Lenovo/Desktop/NoTeach_Weld/College-Student-Welding-Robot-After-Class-Project/aa_robot_work/WeldingRobot_Vision_PathPlanning/01_3D_Models/seam_3_pipe_welding/seam_3_pipe_welding.pcd" ~/ros2_welding_test_data/seam_3_pipe_normalized.pcd
# 用归一化后的点云生成数据集
python3 generate_dataset.py ~/ros2_welding_test_data/seam_3_pipe_normalized.pcd 5 ~/ros2_welding_test_data/pointcloud_dataset/
"""

import sys
import os
import numpy as np
import struct

def read_pcd_file(filename):
    """自动检测并读取 PCD 文件（支持 ASCII 和二进制格式）"""
    
    with open(filename, 'rb') as f:
        content = f.read()
    
    # 尝试解码头部
    try:
        header_text = content.decode('utf-8')
    except:
        # 如果解码失败，尝试用 latin-1
        header_text = content.decode('latin-1')
    
    # 解析头部
    lines = header_text.split('\n')
    fields = []
    sizes = []
    types = []
    counts = []
    width = 0
    height = 0
    points_count = 0
    data_type = None
    data_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            sizes = [int(x) for x in line.split()[1:]]
        elif line.startswith('TYPE'):
            types = line.split()[1:]
        elif line.startswith('COUNT'):
            counts = [int(x) for x in line.split()[1:]]
        elif line.startswith('WIDTH'):
            width = int(line.split()[1])
        elif line.startswith('HEIGHT'):
            height = int(line.split()[1])
        elif line.startswith('POINTS'):
            points_count = int(line.split()[1])
        elif line.startswith('DATA'):
            data_type = line.split()[1]
            # 找到数据开始位置
            data_start = header_text.find('\n', i) + 1
            break
    
    # 重新定位到数据开始处
    data_bytes = content[data_start:]
    
    points = []
    
    if data_type == 'ascii':
        # ASCII 格式
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
        # 二进制格式
        point_step = sum(sizes)  # 每个点的字节数
        for i in range(0, len(data_bytes), point_step):
            if i + point_step <= len(data_bytes):
                point_data = data_bytes[i:i+point_step]
                # 解析 x, y, z (前三个字段，假设都是 float)
                x = struct.unpack('f', point_data[0:4])[0]
                y = struct.unpack('f', point_data[4:8])[0]
                z = struct.unpack('f', point_data[8:12])[0]
                points.append([x, y, z])
    
    elif data_type == 'binary_compressed':
        print("错误: 不支持压缩的二进制格式，请先转换为非压缩格式")
        return np.array([])
    
    print(f"点云点数: {len(points)}")
    return np.array(points)

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
        f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

def main():
    # 参数解析
    if len(sys.argv) < 2:
        print("用法: python3 generate_dataset.py <输入点云.pcd> [帧数] [输出目录]")
        print("示例: python3 generate_dataset.py pointcloud.pcd 10 ./dataset")
        sys.exit(1)
    
    input_file = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./pointcloud_dataset"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)
    
    print(f"读取点云: {input_file}")
    points = read_pcd_file(input_file)
    
    if len(points) == 0:
        print("错误: 无法读取点云数据")
        sys.exit(1)
    
    # 计算点云的中心范围
    center = np.mean(points, axis=0)
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    z_range = np.max(points[:, 2]) - np.min(points[:, 2])
    
    print(f"点云中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"范围: x={x_range:.3f}, y={y_range:.3f}, z={z_range:.3f}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成多帧数据
    for frame in range(1, num_frames + 1):
        # 对点云进行轻微变换（模拟不同视角）
        angle = (frame - 1) * (2 * np.pi / num_frames)
        
        # 旋转矩阵（绕 Y 轴轻微旋转）
        cos_a = np.cos(angle * 0.1)  # 轻微旋转
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
        
        # 保存 PCD 文件
        pcd_file = os.path.join(output_dir, f"{frame}_frame.pcd")
        write_pcd_ascii(pcd_file, transformed)
        
        # 生成相机位置（模拟相机围绕点云旋转）
        camera_radius = max(x_range, y_range, z_range) * 1.5
        camera_x = center[0] + camera_radius * np.cos(angle)
        camera_z = center[2] + camera_radius * np.sin(angle)
        camera_y = center[1] + 0.2
        
        # 保存 TXT 文件
        txt_file = os.path.join(output_dir, f"{frame}_frame.txt")
        with open(txt_file, 'w') as f:
            f.write(f"{camera_x:.6f}\t{camera_y:.6f}\t{camera_z:.6f}\n")
        
        print(f"生成第 {frame} 帧: {pcd_file}")
    
    print(f"\n完成! 生成了 {num_frames} 帧数据")
    print(f"输出目录: {output_dir}")
    print("\n在 main.cpp 中设置 dataset_folder_path:")
    print(f'string dataset_folder_path = "{os.path.abspath(output_dir)}/";')

if __name__ == "__main__":
    main()
    