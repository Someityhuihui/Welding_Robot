#!/usr/bin/env python3
"""
归一化 ASCII 格式的 PCD 点云坐标
用法: python3 normalize_ascii.py <输入.pcd> <输出.pcd>
"""

import sys
import numpy as np

def read_pcd_ascii(filename):
    """读取 ASCII 格式的 PCD 文件"""
    points = []
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # 找到数据开始的位置
    data_start = 0
    for i, line in enumerate(lines):
        if 'DATA ascii' in line:
            data_start = i + 1
            break
    
    # 读取点云数据
    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
    
    print(f"读取到 {len(points)} 个点")
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
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"保存 {len(points)} 个点到 {filename}")

def main():
    if len(sys.argv) < 3:
        print("用法: python3 normalize_ascii.py <输入.pcd> <输出.pcd>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"读取点云: {input_file}")
    points = read_pcd_ascii(input_file)
    
    if len(points) == 0:
        print("错误: 无法读取点云数据")
        sys.exit(1)
    
    print(f"原始坐标范围:")
    print(f"  X: [{np.min(points[:,0]):.2f}, {np.max(points[:,0]):.2f}]")
    print(f"  Y: [{np.min(points[:,1]):.2f}, {np.max(points[:,1]):.2f}]")
    print(f"  Z: [{np.min(points[:,2]):.2f}, {np.max(points[:,2]):.2f}]")
    
    # 平移到原点
    center_x = (np.min(points[:,0]) + np.max(points[:,0])) / 2
    center_y = (np.min(points[:,1]) + np.max(points[:,1])) / 2
    center_z = (np.min(points[:,2]) + np.max(points[:,2])) / 2
    
    points[:,0] -= center_x
    points[:,1] -= center_y
    points[:,2] -= center_z
    
    # 缩放到 [-0.25, 0.25] 范围
    max_range = max(
        np.max(points[:,0]) - np.min(points[:,0]),
        np.max(points[:,1]) - np.min(points[:,1]),
        np.max(points[:,2]) - np.min(points[:,2])
    )
    
    if max_range > 0:
        scale = 0.5 / max_range
        points = points * scale
    
    print(f"\n归一化后坐标范围:")
    print(f"  X: [{np.min(points[:,0]):.4f}, {np.max(points[:,0]):.4f}]")
    print(f"  Y: [{np.min(points[:,1]):.4f}, {np.max(points[:,1]):.4f}]")
    print(f"  Z: [{np.min(points[:,2]):.4f}, {np.max(points[:,2]):.4f}]")
    
    write_pcd_ascii(output_file, points)
    print(f"\n完成! 输出: {output_file}")

if __name__ == "__main__":
    main()
    