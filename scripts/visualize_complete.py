#!/usr/bin/env python3
"""
完整可视化：分割平面 + 角点 + 焊缝 + 相机位置
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import open3d as o3d
import os
import sys

def load_point_cloud(pcd_path, max_points=30000):
    """加载点云并降采样"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    return points

def load_camera_poses(camera_poses_path):
    """加载相机位姿"""
    if not os.path.exists(camera_poses_path):
        print(f"警告: 相机位姿文件不存在: {camera_poses_path}")
        return None
    
    poses = []
    with open(camera_poses_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    poses.append([x, y, z])
                except:
                    continue
    return np.array(poses) if poses else None

def load_colored_planes(planes_dir, max_points_per_plane=5000):
    """加载彩色平面点云"""
    all_points = []
    all_colors = []
    plane_info = []
    
    if not os.path.exists(planes_dir):
        print(f"警告: 平面目录不存在: {planes_dir}")
        return None, None, None
    
    color_list = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80'
    ]
    
    for i in range(100):
        plane_path = os.path.join(planes_dir, f"plane_{i}.pcd")
        if not os.path.exists(plane_path):
            break
        
        pcd = o3d.io.read_point_cloud(plane_path)
        points = np.asarray(pcd.points)
        
        if len(points) > max_points_per_plane:
            indices = np.random.choice(len(points), max_points_per_plane, replace=False)
            points = points[indices]
        
        color_hex = color_list[i % len(color_list)]
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        
        all_points.extend(points)
        all_colors.extend([[r, g, b]] * len(points))
        plane_info.append({'id': i, 'color': color_hex, 'points': len(points)})
    
    if all_points:
        return np.array(all_points), np.array(all_colors), plane_info
    return None, None, None

def load_corners(corners_path):
    """加载角点信息"""
    if not os.path.exists(corners_path):
        print(f"警告: 角点文件不存在: {corners_path}")
        return None
    
    df = pd.read_csv(corners_path)
    return df

def visualize_complete(output_dir, output_html="complete_visualization.html"):
    """完整可视化：分割平面 + 角点 + 焊缝 + 相机位置"""
    
    print("=" * 60)
    print("完整可视化：分割平面 | 角点 | 焊缝 | 相机位置")
    print("=" * 60)
    
    # 加载数据
    csv_path = os.path.join(output_dir, "weld_seams.csv")
    model_pcd_path = os.path.join(output_dir, "global_cloud.pcd")
    planes_dir = os.path.join(output_dir, "planes")
    corners_path = os.path.join(output_dir, "corners_info.txt")
    camera_poses_path = os.path.join(output_dir, "camera_poses.txt")
    
    print(f"\n[1/5] 加载焊缝数据...")
    if os.path.exists(csv_path):
        df_welds = pd.read_csv(csv_path)
        print(f"  焊缝数: {df_welds['weld_id'].nunique()}")
    else:
        df_welds = None
        print("  无焊缝数据")
    
    print(f"\n[2/5] 加载模型点云...")
    model_points = load_point_cloud(model_pcd_path) if os.path.exists(model_pcd_path) else None
    print(f"  模型点云: {len(model_points) if model_points is not None else 0} 点")
    
    print(f"\n[3/5] 加载彩色平面...")
    plane_points, plane_colors, plane_info = load_colored_planes(planes_dir)
    if plane_points is not None:
        print(f"  彩色平面: {len(plane_points)} 点, {len(plane_info)} 个平面")
    else:
        print("  无平面数据")
    
    print(f"\n[4/5] 加载角点...")
    corners_df = load_corners(corners_path)
    if corners_df is not None:
        concave_corners = corners_df[corners_df['is_concave'] == 1]
        convex_corners = corners_df[corners_df['is_concave'] == 0]
        print(f"  角点总数: {len(corners_df)}")
        print(f"    内凹角(焊缝): {len(concave_corners)}")
        print(f"    外凸角(边角): {len(convex_corners)}")
    else:
        concave_corners = convex_corners = None
        print("  无角点数据")
    
    print(f"\n[5/5] 加载相机位置...")
    camera_poses = load_camera_poses(camera_poses_path)
    if camera_poses is not None:
        print(f"  相机位置: {len(camera_poses)} 个")
        # 打印相机范围
        print(f"    相机 X: [{camera_poses[:,0].min():.3f}, {camera_poses[:,0].max():.3f}]")
        print(f"    相机 Y: [{camera_poses[:,1].min():.3f}, {camera_poses[:,1].max():.3f}]")
        print(f"    相机 Z: [{camera_poses[:,2].min():.3f}, {camera_poses[:,2].max():.3f}]")
    else:
        camera_poses = None
        print("  无相机位置数据")
    
    # 创建图形
    fig = go.Figure()
    
    # 1. 模型点云（灰色背景）
    if model_points is not None and len(model_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=model_points[:, 0], y=model_points[:, 1], z=model_points[:, 2],
            mode='markers', name='Model Point Cloud',
            marker=dict(size=1, color='lightgray', opacity=0.3),
            showlegend=True
        ))
    
    # 2. 彩色平面（不同颜色）
    if plane_points is not None and len(plane_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=plane_points[:, 0], y=plane_points[:, 1], z=plane_points[:, 2],
            mode='markers', name='Colored Planes',
            marker=dict(size=1.5, color=plane_colors, opacity=0.6),
            showlegend=True
        ))
    
    # 3. 内凹角（绿色，需要焊接）
    if concave_corners is not None and len(concave_corners) > 0:
        fig.add_trace(go.Scatter3d(
            x=concave_corners['x'], y=concave_corners['y'], z=concave_corners['z'],
            mode='markers', name=f'Concave Corners (Welds) ({len(concave_corners)})',
            marker=dict(size=10, color='green', symbol='diamond', opacity=1.0),
            showlegend=True
        ))
    
    # 4. 外凸角（红色，不焊接）
    if convex_corners is not None and len(convex_corners) > 0:
        fig.add_trace(go.Scatter3d(
            x=convex_corners['x'], y=convex_corners['y'], z=convex_corners['z'],
            mode='markers', name=f'Convex Corners (Edges) ({len(convex_corners)})',
            marker=dict(size=6, color='red', symbol='circle', opacity=0.8),
            showlegend=True
        ))
    
    # 5. 相机位置（蓝色，带连线）
    if camera_poses is not None and len(camera_poses) > 0:
        # 相机位置点
        fig.add_trace(go.Scatter3d(
            x=camera_poses[:, 0], y=camera_poses[:, 1], z=camera_poses[:, 2],
            mode='markers',
            name=f'Camera Positions ({len(camera_poses)})',
            marker=dict(size=5, color='cyan', symbol='circle', opacity=0.9),
            showlegend=True
        ))
        
        # 按顺序连接相机位置（形成轨迹）
        fig.add_trace(go.Scatter3d(
            x=camera_poses[:, 0], y=camera_poses[:, 1], z=camera_poses[:, 2],
            mode='lines',
            name='Camera Trajectory',
            line=dict(color='cyan', width=2, dash='dot'),
            showlegend=True
        ))
        
        # 添加从原点(0,0,0)到相机的参考线（可选）
        # 添加起点和终点标记
        fig.add_trace(go.Scatter3d(
            x=[camera_poses[0, 0]], y=[camera_poses[0, 1]], z=[camera_poses[0, 2]],
            mode='markers',
            name='Camera Start',
            marker=dict(size=8, color='yellow', symbol='square'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[camera_poses[-1, 0]], y=[camera_poses[-1, 1]], z=[camera_poses[-1, 2]],
            mode='markers',
            name='Camera End',
            marker=dict(size=8, color='orange', symbol='square'),
            showlegend=True
        ))
    
    # 6. 焊缝轨迹（彩色线条）
    if df_welds is not None:
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        for weld_id, group in df_welds.groupby('weld_id'):
            color = colors[weld_id % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=group['x'], y=group['y'], z=group['z'],
                mode='lines+markers',
                name=f'Weld {weld_id}',
                line=dict(color=color, width=3),
                marker=dict(size=2, color=color),
                showlegend=True
            ))
    
    # 布局设置
    title_text = f'完整可视化 | 平面: {len(plane_info) if plane_info else 0}'
    title_text += f' | 角点: {len(corners_df) if corners_df is not None else 0}'
    title_text += f' | 焊缝: {df_welds["weld_id"].nunique() if df_welds is not None else 0}'
    title_text += f' | 相机: {len(camera_poses) if camera_poses is not None else 0}'
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20)),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='black'
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, 
                    font=dict(size=10), bgcolor='rgba(0,0,0,0.7)'),
        paper_bgcolor='black',
        plot_bgcolor='black',
        width=1400,
        height=900
    )
    
    output_path = os.path.join(output_dir, output_html)
    fig.write_html(output_path)
    
    print(f"\n✅ 可视化完成！")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return fig

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../output"
    visualize_complete(output_dir)
    