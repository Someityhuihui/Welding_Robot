import numpy as np
import open3d as o3d
import os
import copy

def normalize_mesh_preserve_structure(mesh_path, output_path=None, method='center_and_scale', target_size=2.0):
    """
    归一化网格但保留三角形面片结构
    
    Args:
        mesh_path: 输入网格文件路径（PLY格式）
        output_path: 输出文件路径
        method: 'center_only', 'center_and_scale', 'center_and_unit'
        target_size: 目标尺寸（用于center_and_scale）
    
    Returns:
        normalized_mesh: 归一化后的网格（保留三角形）
        output_path: 输出文件路径
    """
    # 读取网格（保留三角形信息）
    print(f"读取网格: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # 检查是否有三角形
    if len(mesh.triangles) == 0:
        raise ValueError("输入文件没有三角形面片！请确保输入的是网格文件（包含faces）")
    
    print(f"原始网格信息:")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  面数: {len(mesh.triangles)}")
    
    # 获取原始顶点
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    print(f"  原始范围: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
    print(f"            Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
    print(f"            Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
    
    # 计算中心
    center = np.mean(vertices, axis=0)
    
    if method == 'center_only':
        # 只中心化
        vertices_transformed = vertices - center
        transform_info = f"中心化: 平移 {-center}"
        
    elif method == 'center_and_scale':
        # 中心化并缩放到指定尺寸
        vertices_centered = vertices - center
        current_size = np.max(vertices_centered.max(axis=0) - vertices_centered.min(axis=0))
        scale = target_size / current_size
        vertices_transformed = vertices_centered * scale
        transform_info = f"中心化并缩放到 {target_size}: 平移 {-center}, 缩放 {scale:.4f}"
        
    elif method == 'center_and_unit':
        # 中心化并缩放到单位球内
        vertices_centered = vertices - center
        max_distance = np.max(np.linalg.norm(vertices_centered, axis=1))
        scale = 1.0 / max_distance
        vertices_transformed = vertices_centered * scale
        transform_info = f"中心化并缩放到单位球: 平移 {-center}, 缩放 {scale:.4f}"
        
    else:
        raise ValueError(f"未知的归一化方法: {method}")
    
    # 创建新的网格（保留三角形）
    normalized_mesh = o3d.geometry.TriangleMesh()
    normalized_mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed)
    normalized_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 重新计算法向量（用于光照效果）
    normalized_mesh.compute_vertex_normals()
    
    # 如果有颜色信息，也要保留
    if mesh.has_vertex_colors():
        normalized_mesh.vertex_colors = mesh.vertex_colors
        print("  保留了顶点颜色")
    
    if mesh.has_vertex_normals():
        normalized_mesh.vertex_normals = mesh.vertex_normals
        print("  保留了顶点法向量")
    
    print(f"\n归一化后网格信息:")
    print(f"  {transform_info}")
    print(f"  新范围: X[{vertices_transformed[:,0].min():.3f}, {vertices_transformed[:,0].max():.3f}]")
    print(f"          Y[{vertices_transformed[:,1].min():.3f}, {vertices_transformed[:,1].max():.3f}]")
    print(f"          Z[{vertices_transformed[:,2].min():.3f}, {vertices_transformed[:,2].max():.3f}]")
    print(f"  顶点数: {len(normalized_mesh.vertices)}")
    print(f"  面数: {len(normalized_mesh.triangles)}")
    
    # 保存网格（保存为PLY格式以保留三角形）
    if output_path is None:
        output_path = mesh_path.replace('.ply', f'_normalized_{method}.ply')
    
    o3d.io.write_triangle_mesh(output_path, normalized_mesh)
    print(f"\n归一化网格已保存: {output_path}")
    
    # 验证保存的文件
    verify_mesh = o3d.io.read_triangle_mesh(output_path)
    if len(verify_mesh.triangles) > 0:
        print(f"✓ 验证成功: 输出文件包含 {len(verify_mesh.triangles)} 个三角形面片")
    else:
        print(f"✗ 警告: 输出文件似乎丢失了三角形面片")
    
    return normalized_mesh, output_path


def verify_mesh_integrity(mesh_path):
    """验证网格文件的完整性"""
    print(f"\n验证网格文件: {mesh_path}")
    
    # 尝试用Open3D读取
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    print(f"  顶点数: {len(vertices)}")
    print(f"  三角形数: {len(triangles)}")
    print(f"  是否有顶点颜色: {mesh.has_vertex_colors()}")
    print(f"  是否有顶点法向量: {mesh.has_vertex_normals()}")
    
    if len(triangles) == 0:
        print("  ⚠️ 警告: 没有三角形面片！")
        return False
    else:
        print("  ✓ 网格结构完整")
        return True


def batch_normalize_meshes(input_dir, output_dir=None, method='center_and_scale', target_size=2.0):
    """批量归一化文件夹中的所有网格文件"""
    if output_dir is None:
        output_dir = input_dir + "_normalized"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有PLY文件
    ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    
    for ply_file in ply_files:
        input_path = os.path.join(input_dir, ply_file)
        output_path = os.path.join(output_dir, ply_file.replace('.ply', f'_normalized.ply'))
        
        print("\n" + "="*60)
        print(f"处理: {ply_file}")
        print("="*60)
        
        try:
            normalize_mesh_preserve_structure(input_path, output_path, method, target_size)
        except Exception as e:
            print(f"错误处理 {ply_file}: {e}")


def compare_meshes(original_path, normalized_path):
    """对比原始网格和归一化后的网格"""
    # 读取网格
    original = o3d.io.read_triangle_mesh(original_path)
    normalized = o3d.io.read_triangle_mesh(normalized_path)
    
    # 设置颜色
    original.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
    normalized.paint_uniform_color([0.2, 0.6, 0.8])  # 蓝色
    
    # 计算偏移量用于并排显示
    bounds = original.get_axis_aligned_bounding_box()
    offset = (bounds.get_max_bound()[0] - bounds.get_min_bound()[0]) + 1.0
    normalized.translate([offset, 0, 0])
    
    # 可视化对比
    o3d.visualization.draw_geometries(
        [original, normalized],
        window_name="Original (Left) vs Normalized (Right)",
        width=1200,
        height=600
    )


# ============================================================
# 使用示例和测试
# ============================================================

def test_with_sample_mesh():
    """测试网格归一化功能"""
    
    # 创建一个测试网格（立方体）
    print("创建测试网格...")
    test_mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.0, depth=2.0)
    test_mesh.translate([1.0, 0.5, 1.0])
    test_mesh.compute_vertex_normals()
    
    # 添加一些颜色
    colors = np.tile([0.5, 0.5, 0.5], (len(test_mesh.vertices), 1))
    test_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    test_path = "test_mesh.ply"
    o3d.io.write_triangle_mesh(test_path, test_mesh)
    print(f"测试网格已保存: {test_path}")
    print(f"  顶点数: {len(test_mesh.vertices)}")
    print(f"  面数: {len(test_mesh.triangles)}")
    
    # 归一化
    print("\n" + "="*60)
    print("归一化测试网格")
    print("="*60)
    
    normalized_mesh, norm_path = normalize_mesh_preserve_structure(
        test_path, 
        method='center_and_scale',
        target_size=2.0
    )
    
    # 验证
    verify_mesh_integrity(norm_path)
    
    # 对比
    print("\n对比原始和归一化网格...")
    compare_meshes(test_path, norm_path)
    
    return test_path, norm_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='保留网格结构的归一化工具')
    parser.add_argument('--input', '-i', type=str, help='输入网格文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径')
    parser.add_argument('--method', '-m', type=str, default='center_and_scale',
                       choices=['center_only', 'center_and_scale', 'center_and_unit'],
                       help='归一化方法')
    parser.add_argument('--target-size', '-s', type=float, default=2.0,
                       help='目标尺寸（用于center_and_scale）')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--verify', type=str, help='验证网格文件完整性')
    
    args = parser.parse_args()
    
    if args.test:
        # 运行测试
        test_with_sample_mesh()
    
    elif args.verify:
        # 验证网格文件
        verify_mesh_integrity(args.verify)
    
    elif args.input:
        # 归一化网格
        if not os.path.exists(args.input):
            print(f"错误: 文件不存在: {args.input}")
        else:
            normalize_mesh_preserve_structure(
                args.input, 
                args.output, 
                args.method, 
                args.target_size
            )
    else:
        # 显示帮助
        parser.print_help()
        