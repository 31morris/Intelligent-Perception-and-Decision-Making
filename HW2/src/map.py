import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Specify the correct file paths
point_path = '/home/morris/pdm-f24/hw2/semantic_3d_pointcloud/point.npy'
color_path = '/home/morris/pdm-f24/hw2/semantic_3d_pointcloud/color01.npy'  # Or use color0255.npy

# Load the point cloud and color data
points = np.load(point_path)
colors = np.load(color_path)

# Scale correction
points = points * 10000 / 255

# Create Open3D PointCloud object
pcd = o3d.geometry.PointCloud()

# Load points and colors into the PointCloud object
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)  # If using color0255.npy, normalize by dividing by 255.0

# Display the original point cloud
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # Create a coordinate frame
o3d.visualization.draw_geometries([pcd]+[axis], window_name="Original 3D Point Cloud")

###################################################

# Define the y-axis height threshold for the ceiling
ceiling_y = 0.0135

# Filter out points above the ceiling
xyz_points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

filtered_ceiling_xyz_points = xyz_points[xyz_points[:, 1] <= ceiling_y]
filtered_ceiling_colors = colors[xyz_points[:, 1] <= ceiling_y]

# Update the point cloud
filtered_ceiling_pcd = o3d.geometry.PointCloud()
filtered_ceiling_pcd.points = o3d.utility.Vector3dVector(filtered_ceiling_xyz_points)
filtered_ceiling_pcd.colors = o3d.utility.Vector3dVector(filtered_ceiling_colors)

# Display the point cloud after filtering out the ceiling
o3d.visualization.draw_geometries([filtered_ceiling_pcd], window_name="3D Point Cloud After Ceiling Filtering")

####################################################

# Define the y-axis height threshold for the floor
floor_y = -1.35  # Height of the floor

# Filter out points below the floor
filtered_floor_xyz_points = filtered_ceiling_xyz_points[filtered_ceiling_xyz_points[:, 1] >= floor_y]
filtered_floor_colors = filtered_ceiling_colors[filtered_ceiling_xyz_points[:, 1] >= floor_y]

# Update the point cloud
filtered_floor_pcd = o3d.geometry.PointCloud()
filtered_floor_pcd.points = o3d.utility.Vector3dVector(filtered_floor_xyz_points)
filtered_floor_pcd.colors = o3d.utility.Vector3dVector(filtered_floor_colors)

# Display the point cloud after filtering out the floor
o3d.visualization.draw_geometries([filtered_floor_pcd], window_name="3D Point Cloud After Floor Filtering")

####################################################

# Save the filtered XYZ coordinates and color data
np.save('/home/morris/pdm-f24/hw2/filtered_points.npy', np.asarray(filtered_floor_pcd.points))
np.save('/home/morris/pdm-f24/hw2/filtered_colors.npy', np.asarray(filtered_floor_pcd.colors))
print('Filtered point cloud data has been successfully saved')

####################################################
# Draw a scatter plot, adjust image size and format to match the first section
pixel_per_inches = 1 / plt.rcParams['figure.dpi']
plt.figure(figsize=(1700 * pixel_per_inches, 1100 * pixel_per_inches))  # Adjust figure size

plt.scatter(filtered_floor_xyz_points[:, 2], filtered_floor_xyz_points[:, 0], 
            c=filtered_floor_colors, s=5)  # Set point size and color
plt.axis('off')  # Turn off the axes

# Save the image in the same format as the first section
plt.savefig('/home/morris/pdm-f24/hw2/map.png', bbox_inches='tight', pad_inches=0)  # Save the image, removing margins
plt.show()  # Display the image
print('finish')
