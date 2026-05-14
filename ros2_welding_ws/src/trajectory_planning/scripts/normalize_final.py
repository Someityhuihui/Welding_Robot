#!/usr/bin/env python3
import sys
import numpy as np

def read_pcd_ascii(filename):
    points = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 找到数据开始位置
    data_start = 0
    for i, line in enumerate(lines):
        if 'DATA ascii' in line:
            data_start = i + 1
            break
    
    # 读取点云
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
                except:
                    continue
    
    print(f"读取 {len(points)} 个点")
    return np.array(points)

def write_pcd_ascii(filename, points):
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

if len(sys.argv) < 3:
    print("用法: python3 normalize_final.py <输入.pcd> <输出.pcd>")
    sys.exit(1)

# 读取
points = read_pcd_ascii(sys.argv[1])
if len(points) == 0:
    print("错误: 无数据")
    sys.exit(1)

print(f"原始范围:")
print(f"  X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
print(f"  Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
print(f"  Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")

# 归一化：平移到中心，缩放到合适大小
center = (points.min(axis=0) + points.max(axis=0)) / 2
points -= center

max_range = max(
    points[:,0].max() - points[:,0].min(),
    points[:,1].max() - points[:,1].min(),
    points[:,2].max() - points[:,2].min()
)
scale = 0.5 / max_range
points *= scale

print(f"\n归一化后范围:")
print(f"  X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
print(f"  Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
print(f"  Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")

write_pcd_ascii(sys.argv[2], points)
print(f"\n保存到: {sys.argv[2]}")
