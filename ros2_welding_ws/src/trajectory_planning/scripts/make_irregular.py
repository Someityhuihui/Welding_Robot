#!/usr/bin/env python3
"""
将规则点云转换为不规则点云
步骤:
  1. 添加高斯噪声（模拟扫描误差）
  2. 模拟视角相关密度（离相机近的点多，远的点少）
用法: python3 make_irregular.py <输入.pcd> <输出.pcd> [噪声级别] [相机位置]
示例: python3 make_irregular.py input.pcd output.pcd 0.0005 "0.5,0.2,0.5"
"""

import sys
import os
import numpy as np
import random

def read_pcd_ascii(filename):
    """读取 ASCII 格式的 PCD 文件"""
    points = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 找到数据开始位置
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
    print(f"保存 {len(points)} 个点到 {filename}")

def add_gaussian_noise(points, noise_level=0.0005):
    """
    步骤1: 添加高斯噪声（模拟扫描误差）
    noise_level: 噪声标准差（米），建议 0.0001 ~ 0.001
    """
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise
    
    print(f"\n[步骤1] 添加高斯噪声 (sigma={noise_level})")
    print(f"  原始点数: {len(points)}")
    print(f"  噪声后点数: {len(noisy_points)}")
    
    return noisy_points

def simulate_camera_view(points, camera_pos=(0.5, 0, 0.5), density_factor=0.7):
    """
    步骤2: 模拟相机视角（离相机近的点保留率高）
    camera_pos: 相机位置 (x, y, z)
    density_factor: 密度变化因子，0.3~0.9，越大变化越明显
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
    # 距离越近，概率越高；距离越远，概率越低
    probabilities = []
    for d in distances:
        # 归一化距离 0~1
        if dist_range > 0:
            dist_norm = (d - min_dist) / dist_range
        else:
            dist_norm = 0
        # 保留概率 = 1 - density_factor * dist_norm
        # density_factor 越大，远处保留的点越少
        prob = 1.0 - density_factor * dist_norm
        # 限制在 0.1 ~ 0.95 之间
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

def main():
    # 参数解析
    if len(sys.argv) < 3:
        print("用法: python3 make_irregular.py <输入.pcd> <输出.pcd> [选项]")
        print("\n选项:")
        print("  --noise <值>      高斯噪声标准差 (默认: 0.0005)")
        print("  --camera <x,y,z>  相机位置 (默认: 0.5,0,0.5)")
        print("  --density <值>    密度变化因子 0.3~0.9 (默认: 0.7)")
        print("\n示例:")
        print("  python3 make_irregular.py input.pcd output.pcd")
        print("  python3 make_irregular.py input.pcd output.pcd --noise 0.001 --camera 0.3,0.2,0.4")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 默认参数
    noise_level = 0.0005
    camera_pos = (0.5, 0, 0.5)
    density_factor = 0.7
    
    # 解析可选参数
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--noise' and i+1 < len(sys.argv):
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
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"噪声级别: {noise_level}")
    print(f"相机位置: {camera_pos}")
    print(f"密度因子: {density_factor}")
    print("-" * 50)
    
    # 读取点云
    points = read_pcd_ascii(input_file)
    
    if len(points) == 0:
        print("错误: 无法读取点云数据")
        sys.exit(1)
    
    print(f"\n原始点云范围:")
    print(f"  X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
    print(f"  Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
    print(f"  Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")
    
    # 步骤1: 添加高斯噪声
    points = add_gaussian_noise(points, noise_level)
    
    # 步骤2: 模拟相机视角密度
    points = simulate_camera_view(points, camera_pos, density_factor)
    
    # 保存结果
    write_pcd_ascii(output_file, points)
    
    print("\n" + "=" * 50)
    print(f"完成! 不规则点云已保存到: {output_file}")
    print("=" * 50)

if __name__ == "__main__":
    main()

