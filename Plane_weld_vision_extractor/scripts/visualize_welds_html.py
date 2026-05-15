#!/usr/bin/env python3
# visualize_complete.py - 完整可视化：模型点云 + 彩色平面 + 焊缝轨迹
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import open3d as o3d
import os
import sys

def load_point_cloud(pcd_path, max_points=50000, color=None):
    """加载点云并降采样，可选着色"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    # 如果点太多，随机采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        
        # 如果有颜色信息，也要采样
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[indices]
            return points, colors
    
    if pcd.has_colors():
        return points, np.asarray(pcd.colors)
    
    return points, None

def load_colored_planes(planes_dir, max_points_per_plane=10000):
    """加载每个平面的彩色点云"""
    all_points = []
    all_colors = []
    plane_info = []
    
    if not os.path.exists(planes_dir):
        print(f"警告: 平面目录不存在: {planes_dir}")
        return None, None, None
    
    # 预定义颜色列表
    color_list = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80'
    ]
    
    for i in range(100):  # 最多100个平面
        plane_path = os.path.join(planes_dir, f"plane_{i}.pcd")
        if not os.path.exists(plane_path):
            break
        
        pcd = o3d.io.read_point_cloud(plane_path)
        points = np.asarray(pcd.points)
        
        # 降采样
        if len(points) > max_points_per_plane:
            indices = np.random.choice(len(points), max_points_per_plane, replace=False)
            points = points[indices]
        
        # 使用固定颜色
        color_hex = color_list[i % len(color_list)]
        # 转换为RGB 0-1范围
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        
        all_points.extend(points)
        all_colors.extend([[r, g, b]] * len(points))
        plane_info.append({
            'id': i,
            'color': color_hex,
            'points': len(points)
        })
    
    if all_points:
        return np.array(all_points), np.array(all_colors), plane_info
    return None, None, None

def visualize_complete(csv_path, model_pcd_path, planes_dir=None, output_html="complete_visualization.html"):
    """完整可视化：模型点云 + 彩色平面 + 焊缝轨迹"""
    
    print("=" * 60)
    print("焊缝系统完整可视化")
    print("=" * 60)
    
    # 1. 加载焊缝数据
    print(f"\n[1/4] 加载焊缝数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"错误: 找不到焊缝文件 {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"  焊缝数: {df['weld_id'].nunique()}, 总点数: {len(df)}")
    
    # 2. 加载模型点云
    print(f"\n[2/4] 加载模型点云: {model_pcd_path}")
    model_points, model_colors = load_point_cloud(model_pcd_path, max_points=30000)
    print(f"  模型点云: {len(model_points)} 个点")
    
    # 3. 加载彩色平面
    print(f"\n[3/4] 加载彩色平面: {planes_dir}")
    plane_points, plane_colors, plane_info = load_colored_planes(planes_dir, max_points_per_plane=8000)
    if plane_points is not None:
        print(f"  彩色平面: {len(plane_points)} 个点, {len(plane_info)} 个平面")
        for info in plane_info:
            print(f"    平面 {info['id']}: {info['points']} 点, 颜色 {info['color']}")
    else:
        print("  未找到平面数据，将只显示模型和焊缝")
    
    # 4. 创建图形
    print(f"\n[4/4] 生成HTML可视化...")
    fig = go.Figure()
    
    # 添加模型点云（灰色，半透明）
    if len(model_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=model_points[:, 0],
            y=model_points[:, 1],
            z=model_points[:, 2],
            mode='markers',
            name='Model Point Cloud',
            marker=dict(
                size=1,
                color='lightgray',
                opacity=0.3,
                symbol='circle'
            ),
            showlegend=True
        ))
    
    # 添加彩色平面点云
    if plane_points is not None and len(plane_points) > 0:
        # 按平面分组显示（每个平面单独显示，便于控制）
        fig.add_trace(go.Scatter3d(
            x=plane_points[:, 0],
            y=plane_points[:, 1],
            z=plane_points[:, 2],
            mode='markers',
            name='Colored Planes',
            marker=dict(
                size=1.5,
                color=plane_colors,
                opacity=0.7,
                symbol='circle'
            ),
            showlegend=True
        ))
    
    # 添加焊缝轨迹（彩色线条）
    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80',
        '#800000', '#008000', '#000080', '#808000', '#800080', '#008080'
    ]
    
    for weld_id, group in df.groupby('weld_id'):
        x = group['x'].values
        y = group['y'].values
        z = group['z'].values
        color = colors[weld_id % len(colors)]
        
        # 计算长度
        length = 0
        for i in range(1, len(x)):
            length += np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=f'Weld {weld_id} ({len(x)} pts, {length*1000:.1f}mm)',
            line=dict(color=color, width=4),
            marker=dict(size=2, color=color),
            showlegend=True
        ))
    
    # 设置布局
    title_text = f'焊缝系统完整可视化'
    if plane_info:
        title_text += f' | 平面数: {len(plane_info)}'
    title_text += f' | 焊缝数: {df["weld_id"].nunique()}'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='black'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0.7)'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=1400,
        height=900
    )
    
    # 保存为 HTML
    fig.write_html(output_html)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)
    print(f"输出文件: {output_html}")
    print(f"文件大小: {os.path.getsize(output_html) / 1024 / 1024:.2f} MB")
    print("\n内容统计:")
    print(f"  - 模型点云: {len(model_points):,} 点")
    if plane_points is not None:
        print(f"  - 彩色平面: {len(plane_points):,} 点 ({len(plane_info)} 个平面)")
    print(f"  - 焊缝轨迹: {df['weld_id'].nunique()} 条, {len(df):,} 个点")
    print("\n用浏览器打开即可查看、旋转、缩放")
    print("=" * 60)
    
    return fig

def main():
    # 设置路径 - 根据你的输出目录修改
    base_path = "../seam_fillet_lap_scene_normalized_irregular_output"
    
    # 如果指定了命令行参数
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    
    csv_path = os.path.join(base_path, "weld_seams.csv")
    model_pcd_path = os.path.join(base_path, "global_cloud.pcd")
    planes_dir = os.path.join(base_path, "planes")
    output_html = os.path.join(base_path, "complete_visualization.html")
    
    print(f"数据目录: {base_path}")
    print(f"焊缝文件: {csv_path}")
    print(f"模型点云: {model_pcd_path}")
    print(f"平面目录: {planes_dir}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 找不到焊缝文件 {csv_path}")
        sys.exit(1)
    
    if not os.path.exists(model_pcd_path):
        print(f"警告: 找不到模型点云 {model_pcd_path}")
        model_pcd_path = None
    
    visualize_complete(csv_path, model_pcd_path, planes_dir, output_html)

if __name__ == "__main__":
    main()
    