import numpy as np
import open3d as o3d
import os
import time

"""
Args:
    mesh_path: 网格文件路径（必须是包含三角形面片的网格）
    output_dir: 输出目录
    num_frames: 扫描帧数
    radius: 扫描半径
    height: 扫描高度
    num_rays_h: 水平射线数
    num_rays_v: 垂直射线数
    fov_v: 垂直视场角（度）
    max_range: 最大扫描距离
    rotation_axis: 旋转轴 ('X', 'Y', 'Z')
"""

# 使用归一化后的网格直接扫描
# python generate_pointcloud_data.py --input seam_fillet_lap_scene_normalized_center_and_scale.ply --output ./scan_result --frames 36 --radius 3.0 --rotation-axis Z --height 2 --rays-h 512 --rays-v 256


# # 如果你已经有网格文件，直接扫描
# python generate_pointcloud_data.py --input mesh.ply --frames 36 --radius 2.0

# # 快速测试
# python generate_pointcloud_data.py --input mesh.ply --frames 4 --rays-h 90 --rays-v 16

# # 绕不同轴扫描
# python generate_pointcloud_data.py --input mesh.ply --frames 36 --radius 2.0 --rotation-axis X
# python generate_pointcloud_data.py --input mesh.ply --frames 36 --radius 2.0 --rotation-axis Y
# python generate_pointcloud_data.py --input mesh.ply --frames 36 --radius 2.0 --rotation-axis Z


# ============================================================
# 注意：这个文件直接使用网格，不进行泊松重建
# ============================================================

class Open3DLiDARScanner:
    """使用Open3D的RaycastingScene进行加速扫描"""
    
    def __init__(self, mesh_path, num_rays_h=360, num_rays_v=64, 
                 fov_v=30.0, max_range=5.0, verbose=True):
        """
        Args:
            mesh_path: 网格文件路径（必须包含三角形面片）
            num_rays_h: 水平方向射线数
            num_rays_v: 垂直方向射线数
            fov_v: 垂直视场角（度）
            max_range: 最大扫描距离
            verbose: 是否打印详细信息
        """
        print(f"加载网格: {mesh_path}")
        
        # 直接加载网格（不进行任何重建）
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # 检查是否有三角形
        if len(self.mesh.triangles) == 0:
            raise ValueError(f"网格没有三角形面片！请确保输入的是网格文件（包含faces）")
        
        print(f"网格信息: {len(self.mesh.vertices)} 顶点, {len(self.mesh.triangles)} 面")
        
        # 创建RaycastingScene
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene.add_triangles(mesh_t)
        
        self.num_rays_h = num_rays_h
        self.num_rays_v = num_rays_v
        self.total_rays = num_rays_h * num_rays_v
        self.fov_v = np.radians(fov_v)
        self.max_range = max_range
        self.verbose = verbose
        
        if verbose:
            print(f"扫描配置:")
            print(f"  水平射线数: {num_rays_h}")
            print(f"  垂直射线数: {num_rays_v}")
            print(f"  总射线数: {self.total_rays:,}")
            print(f"  垂直视场角: {fov_v}°")
            print(f"  最大距离: {max_range}m")
        
    def scan_from_pose(self, camera_pos, look_at=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
        """
        从给定位置执行扫描
        """
        camera_pos = np.asarray(camera_pos, dtype=np.float32)
        look_at = np.asarray(look_at, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        
        # 构建局部坐标系
        z_axis = look_at - camera_pos
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-10)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # 旋转矩阵
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # 生成射线方向
        rays_d = []
        
        # 水平角
        h_angles = np.linspace(-np.pi, np.pi, self.num_rays_h, endpoint=False)
        # 垂直角
        v_angles = np.linspace(-self.fov_v/2, self.fov_v/2, self.num_rays_v)
        
        for v_angle in v_angles:
            cos_v = np.cos(v_angle)
            sin_v = np.sin(v_angle)
            for h_angle in h_angles:
                # 局部方向
                dir_local = np.array([
                    cos_v * np.sin(h_angle),
                    sin_v,
                    cos_v * np.cos(h_angle)
                ])
                # 全局方向
                dir_global = R @ dir_local
                dir_global = dir_global / (np.linalg.norm(dir_global) + 1e-10)
                rays_d.append(dir_global)
        
        rays_d = np.array(rays_d, dtype=np.float32)
        rays_o = np.tile(camera_pos, (len(rays_d), 1))
        
        # 构建rays tensor: shape (N, 6)
        rays = np.concatenate([rays_o, rays_d], axis=1)
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        # 批量射线投射
        ans = self.scene.cast_rays(rays_tensor)
        
        # 获取交点
        t_hit = ans['t_hit'].numpy()
        hit_mask = (t_hit > 0) & (t_hit < self.max_range)
        
        # 计算交点坐标
        hit_points = rays_o[hit_mask] + rays_d[hit_mask] * t_hit[hit_mask, np.newaxis]
        
        if self.verbose:
            hit_percentage = np.sum(hit_mask) / len(rays_d) * 100
            print(f"  射线: {len(rays_d):,} | 击中: {np.sum(hit_mask):,} ({hit_percentage:.1f}%)")
        
        return hit_points


class CircularScanSimulator:
    """环绕扫描模拟器 - 直接使用网格，不进行重建"""
    
    def __init__(self, mesh_path, output_dir, num_frames=36, radius=2.0, height=1.5,
                 num_rays_h=180, num_rays_v=32, fov_v=40.0, max_range=5.0,
                 rotation_axis='Y'):
        
        self.mesh_path = mesh_path
        self.output_dir = output_dir
        self.num_frames = num_frames
        self.radius = radius
        self.height = height
        self.rotation_axis = rotation_axis.upper()
        
        if self.rotation_axis not in ['X', 'Y', 'Z']:
            raise ValueError(f"rotation_axis 必须是 'X', 'Y' 或 'Z'，得到: {rotation_axis}")
        
        # 初始化扫描器（直接使用网格）
        self.scanner = Open3DLiDARScanner(
            mesh_path,
            num_rays_h=num_rays_h,
            num_rays_v=num_rays_v,
            fov_v=fov_v,
            max_range=max_range,
            verbose=True
        )
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_trajectory(self):
        """生成环绕轨迹（根据选择的旋转轴）"""
        self.camera_positions = []
        
        for i in range(self.num_frames):
            angle = 2 * np.pi * i / self.num_frames
            
            if self.rotation_axis == 'Y':
                # 绕Y轴旋转：在XZ平面，Y固定为height
                x = self.radius * np.cos(angle)
                z = self.radius * np.sin(angle)
                y = self.height
                pos = np.array([x, y, z])
                
            elif self.rotation_axis == 'X':
                # 绕X轴旋转：在YZ平面，X固定为radius
                # 注意：radius作为到X轴的距离
                y = self.radius * np.cos(angle)
                z = self.radius * np.sin(angle)
                x = self.height
                pos = np.array([x, y, z])
                
            elif self.rotation_axis == 'Z':
                # 绕Z轴旋转：在XY平面，Z固定为height
                x = self.radius * np.cos(angle)
                y = self.radius * np.sin(angle)
                z = self.height
                pos = np.array([x, y, z])
            
            self.camera_positions.append(pos)
        
        self.camera_positions = np.array(self.camera_positions)
        return self.camera_positions
    
    def get_look_at_center(self):
        """根据旋转轴获取相机朝向的中心点"""
        if self.rotation_axis == 'Y':
            return np.array([0, 0, 0])
        elif self.rotation_axis == 'X':
            return np.array([0, 0, 0])
        elif self.rotation_axis == 'Z':
            return np.array([0, 0, 0])
    
    def get_up_direction(self):
        """根据旋转轴获取相机的上方向"""
        if self.rotation_axis == 'Y':
            return np.array([0, 1, 0])
        elif self.rotation_axis == 'X':
            # 绕X轴旋转时，上方向为Z轴
            return np.array([0, 0, 1])
        elif self.rotation_axis == 'Z':
            # 绕Z轴旋转时，上方向为Y轴
            return np.array([0, 1, 0])
    
    def simulate_scan(self):
        """执行扫描并保存结果"""
        self.generate_trajectory()
        
        all_scanned_points = []
        
        print(f"\n旋转模式: 绕{self.rotation_axis}轴旋转")
        print(f"半径: {self.radius}, 高度偏移: {self.height}")
        
        for i in range(self.num_frames):
            print(f"\n处理第 {i+1}/{self.num_frames} 帧...")
            
            cam_pos = self.camera_positions[i]
            print(f"  雷达位置: [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
            
            start_time = time.time()
            
            # 执行扫描
            scanned_global = self.scanner.scan_from_pose(
                cam_pos, 
                look_at=self.get_look_at_center(),
                up=self.get_up_direction()
            )
            
            elapsed = time.time() - start_time
            print(f"  耗时: {elapsed:.1f}秒, 点数: {len(scanned_global)}")
            
            if len(scanned_global) > 0:
                # 保存点云为PCD格式（全局坐标系）
                pcd_frame = o3d.geometry.PointCloud()
                pcd_frame.points = o3d.utility.Vector3dVector(scanned_global)
                
                # 保存为PCD文件
                frame_path = os.path.join(self.output_dir, f"{i+1}_frame.pcd")
                o3d.io.write_point_cloud(frame_path, pcd_frame)
                
                # 保存雷达位姿（逗号分隔的x,y,z）
                pose_path = os.path.join(self.output_dir, f"{i+1}_frame.txt")
                with open(pose_path, 'w') as f:
                    f.write(f"{cam_pos[0]:.6f},{cam_pos[1]:.6f},{cam_pos[2]:.6f}")
                
                all_scanned_points.append(scanned_global)
            else:
                print(f"  警告: 第{i+1}帧未扫描到点")
        
        # 保存合并后的点云
        if len(all_scanned_points) > 0:
            merged_points = np.vstack(all_scanned_points)
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
            
            merged_path = os.path.join(self.output_dir, "merged_global.pcd")
            o3d.io.write_point_cloud(merged_path, merged_pcd)
            print(f"\n合并全局点云已保存: {merged_path} ({len(merged_points)} 点)")
        
        # 保存所有相机位姿汇总
        poses_path = os.path.join(self.output_dir, "all_camera_poses.txt")
        with open(poses_path, 'w') as f:
            f.write(f"# Rotation Axis: {self.rotation_axis}\n")
            f.write(f"# Radius: {self.radius}, Height Offset: {self.height}\n")
            for i, pos in enumerate(self.camera_positions):
                f.write(f"frame_{i+1}: {pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")
        
        print(f"\n所有文件已保存到: {self.output_dir}")
        
        return all_scanned_points


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='基于网格的LiDAR扫描模拟（直接使用网格，不进行重建）')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入网格文件路径 (.ply, 必须包含三角形面片)')
    parser.add_argument('--output', '-o', type=str, default='./scan_result',
                       help='输出目录')
    
    # 扫描参数
    parser.add_argument('--frames', '-f', type=int, default=36,
                       help='扫描帧数 (默认: 36)')
    parser.add_argument('--radius', '-r', type=float, default=2.0,
                       help='扫描半径 (默认: 2.0)')
    parser.add_argument('--height', '-H', type=float, default=1.2,
                       help='扫描高度偏移 (默认: 1.2)')
    
    # 旋转轴选择
    parser.add_argument('--rotation-axis', '-a', type=str, default='Y',
                       choices=['X', 'Y', 'Z'],
                       help='旋转轴: X, Y, Z (默认: Y)')
    
    # 射线参数
    parser.add_argument('--rays-h', '-rh', type=int, default=512,
                       help='水平方向射线数 (默认: 512)')
    parser.add_argument('--rays-v', '-rv', type=int, default=256,
                       help='垂直方向射线数 (默认: 256)')
    parser.add_argument('--fov', type=float, default=40.0,
                       help='垂直视场角(度) (默认: 40)')
    parser.add_argument('--max-range', type=float, default=5.0,
                       help='最大扫描距离(米) (默认: 5.0)')
    
    # 其他选项
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='完成后可视化结果')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        return
    
    # 验证输入文件是否包含三角形
    print(f"验证输入网格: {args.input}")
    test_mesh = o3d.io.read_triangle_mesh(args.input)
    if len(test_mesh.triangles) == 0:
        print(f"错误: 输入文件不包含三角形面片！")
        print(f"请先使用网格格式的文件（包含faces），或者使用 normalize_mesh.py 转换点云为网格")
        return
    print(f"✓ 网格验证通过: {len(test_mesh.triangles)} 个三角形面片")
    
    # 计算总射线数
    total_rays_per_frame = args.rays_h * args.rays_v
    print(f"\n扫描配置:")
    print(f"  输入网格: {args.input}")
    print(f"  旋转轴: {args.rotation_axis}")
    print(f"  水平射线: {args.rays_h}")
    print(f"  垂直射线: {args.rays_v}")
    print(f"  每帧总射线: {total_rays_per_frame:,}")
    print(f"  帧数: {args.frames}")
    print(f"  总体射线数: {total_rays_per_frame * args.frames:,}")
    
    print("\n" + "="*60)
    print("执行LiDAR扫描模拟（直接使用网格，无重建）")
    print("="*60)
    
    # 执行扫描
    simulator = CircularScanSimulator(
        mesh_path=args.input,
        output_dir=args.output,
        num_frames=args.frames,
        radius=args.radius,
        height=args.height,
        num_rays_h=args.rays_h,
        num_rays_v=args.rays_v,
        fov_v=args.fov,
        max_range=args.max_range,
        rotation_axis=args.rotation_axis
    )
    
    simulator.simulate_scan()
    
    print("\n完成!")


if __name__ == "__main__":
    main()
    