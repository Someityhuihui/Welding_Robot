import open3d as o3d

# 用户配置参数（允许改动）
root_path = "C:/Users/Lenovo/Desktop/NoTeach_Weld/Welding_Robot/WeldingRobot_Vision_PathPlanning/01_3D_Models/"
scene_name = "seam_fillet_lap_scene"  # 避免中文字符，请使用英文


# 代码执行区（禁止未经许可的改动）
input_file_path = root_path + scene_name + "/" + scene_name + ".ply"
print(f"正在处理文件: {input_file_path}")  # 输出正在处理的文件路径
output_file_path = input_file_path.replace(".ply", ".pcd")  # 将输入文件路径中的 .ply 替换为 .pcd

# 读取 PLY 文件
pcd = o3d.io.read_point_cloud(input_file_path)
# 保存为 PCD 文件
# o3d.io.write_point_cloud("output.pcd", pcd)
# 保存为 ASCII 格式的 PCD
o3d.io.write_point_cloud(output_file_path, pcd, write_ascii=True)
print(f"点数: {len(pcd.points)}")
print("保存pcd的位置在于{}(ASCII 格式)".format(output_file_path))


