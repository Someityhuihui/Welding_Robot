import open3d as o3d

# 读取 PLY 文件
pcd = o3d.io.read_point_cloud("C:/Users/Lenovo/Desktop/NoTeach_Weld/College-Student-Welding-Robot-After-Class-Project/aa_robot_work/WeldingRobot_Vision_PathPlanning/01_3D_Models/seam_3_pipe_welding/seam_3_pipe_welding.ply")

# 保存为 PCD 文件
# o3d.io.write_point_cloud("output.pcd", pcd)

# 保存为 ASCII 格式的 PCD
o3d.io.write_point_cloud("output_ascii.pcd", pcd, write_ascii=True)

print(f"点数: {len(pcd.points)}")
print("保存为 output_ascii.pcd (ASCII 格式)")


