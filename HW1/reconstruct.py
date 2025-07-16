import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import argparse
import os
import copy
import time
import math

def depth_image_to_point_cloud(rgb, depth):
    # Camera intrinsics
    principal_point = np.array([256, 256])
    f = (256) * (1 / np.tan(np.deg2rad(90 / 2)))

    images = [rgb, depth]
    processed_images = []
    
    for image_path in images:
        if image_path == rgb:
            image = np.asarray(o3d.io.read_image(image_path)) / 255  # Normalize RGB values
            processed_images.append(image)
        else:
            depth_image = np.asarray(o3d.io.read_image(image_path)) / 1000  # Convert depth to meters
            processed_images.append(depth_image)

    image, depth = processed_images

    # Generate pixel coordinates
    height, width = depth.shape
    pixel_coords = np.mgrid[0:height, 0:width]  # Create a grid of pixel coordinates

    # Flatten depth and pixel coordinates for processing
    d = depth.flatten()
    x = (principal_point[0] - pixel_coords[1].flatten()) * d / f
    y = (principal_point[1] - pixel_coords[0].flatten()) * d / f
    z = d

    # Stack coordinates into a single array
    xyz_points = np.array((x, y, z)).reshape(3, -1).T

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3))

    # Create camera point
    camera_pcd = o3d.geometry.PointCloud()
    camera_pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])
    camera_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Set the color to red

    return pcd, camera_pcd

def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh          

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])  

def compute_best_fit_transform(src_pts, tgt_pts):
    # Ensure that the dimensions of source and target points match
    assert src_pts.shape == tgt_pts.shape, "Source and target point shapes do not match."

    # Get the number of dimensions
    dimensions = src_pts.shape[1]

    # Calculate the centroid of each set of points
    mean_src = np.mean(src_pts, axis=0)
    mean_tgt = np.mean(tgt_pts, axis=0)

    # Shift the points to their respective centroids
    shifted_src = np.empty_like(src_pts)
    shifted_tgt = np.empty_like(tgt_pts)

    for i in range(src_pts.shape[0]):
        shifted_src[i] = src_pts[i] - mean_src
        shifted_tgt[i] = tgt_pts[i] - mean_tgt

    # Compute the covariance matrix
    covariance_matrix = np.dot(shifted_tgt.T, shifted_src)

    # Use Singular Value Decomposition to get the rotation matrix
    U, singular_values, Vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.dot(U, Vt)

    # Handle the case of reflection
    if np.linalg.det(rotation_matrix) < 0:
        Vt[dimensions - 1, :] *= -1
        rotation_matrix = np.dot(U, Vt)

    # Compute the translation vector
    translation_vector = mean_tgt - np.dot(rotation_matrix, mean_src)

    # Construct the homogeneous transformation matrix
    transform_matrix = np.zeros((dimensions + 1, dimensions + 1))
    for d in range(dimensions + 1):
        transform_matrix[d, d] = 1  # Set the diagonal to 1
    transform_matrix[:dimensions, :dimensions] = rotation_matrix
    transform_matrix[:dimensions, dimensions] = translation_vector

    return transform_matrix, rotation_matrix, translation_vector

def find_nearest_neighbor(src_points, dst_points):
    # Initialize the nearest neighbor searcher
    neighbor_search = NearestNeighbors(n_neighbors=1)
    neighbor_search.fit(dst_points)

    # Compute the nearest neighbor distances and indices for each source point
    distances, indices = neighbor_search.kneighbors(src_points, return_distance=True)

    # Select valid neighbors whose distances are less than 80% of the median
    threshold = np.median(distances) * 0.8
    valid_neighbors = distances < threshold

    return distances[valid_neighbors].ravel(), indices[valid_neighbors].ravel(), valid_neighbors.ravel()

def my_icp(source_down, target_down, trans_init=None, max_iterations=100000, tolerance=0.000005): 
    # the user may turn the parameter

    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    m = np.shape(source_points)[1]

    # Create homogeneous coordinates and copy original points
    src = np.ones((m + 1, source_points.shape[0]))
    dst = np.ones((m + 1, target_points.shape[0]))
    src[0:m, :] = np.copy(source_points.T)
    dst[0:m, :] = np.copy(target_points.T)

    # Initial pose estimation
    if trans_init is not None:
        src = np.dot(trans_init, src)
    
    prev_error = float('inf')  # Initialize previous error to a high value

    # Main iteration loop
    for i in range(max_iterations):
        # Find nearest neighbors between current source and target points
        distances, indices, valid = find_nearest_neighbor(src[0:m, :].T, dst[0:m, :].T)

        # Compute the best-fit transformation between the source and the target
        transformation, _, _ = compute_best_fit_transform(src[0:m, valid].T, dst[0:m, indices].T)

        # Update the source points with the computed transformation
        src = np.dot(transformation, src)

        # Calculate the mean error to check for convergence
        mean_error = np.sum(distances) / np.count_nonzero(distances)
        
        # Check for convergence based on tolerance
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Final transformation calculation
    transformation, _, _ = compute_best_fit_transform(source_points, src[0:m, :].T)

    return transformation

def accumulate_point_clouds(base_cloud, additional_clouds):
    for cloud in additional_clouds:
        # Add points from the new point cloud to the base point cloud
        base_cloud += cloud
    return base_cloud

def load_pose_data(args):
    # Set the file path
    path_to_file = f"{args.data_root}/GT_pose.npy"
    pose_data = np.load(path_to_file)

    # Convert the data into actual coordinates (units: meters)
    if args.floor == 1:
        # Convert millimeters to meters, rw => 10 / 0.25
        x_coords = -pose_data[:, 0] / 40
        y_coords = pose_data[:, 1] / 40
        z_coords = -pose_data[:, 2] / 40
    elif args.floor == 2:
        # Convert the coordinates to meters, rw => 10 / 0.25
        x_coords = -pose_data[:, 0] / 40 - 0.00582
        y_coords = (pose_data[:, 1] / 40) - 0.07313
        z_coords = -pose_data[:, 2] / 40 - 0.03

    # Stack the coordinate points
    point_cloud_data = np.vstack((x_coords, y_coords, z_coords)).T

    # Create a point cloud object
    ground_truth_pcd = o3d.geometry.PointCloud()
    ground_truth_pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    ground_truth_pcd.paint_uniform_color([0, 0, 0])  # Set the color to black

    # Create line segments connecting points
    line_segments = [[i, i + 1] for i in range(len(point_cloud_data) - 1)]

    # Create a line set object
    ground_truth_lines = o3d.geometry.LineSet()
    ground_truth_lines.points = o3d.utility.Vector3dVector(point_cloud_data)
    ground_truth_lines.lines = o3d.utility.Vector2iVector(line_segments)

    return ground_truth_pcd, ground_truth_lines

def reconstruct(args):

    # config
    voxel_size = 0.00225
    point_cloud = o3d.geometry.PointCloud()
    estimate_camera_cloud = o3d.geometry.PointCloud()
    data_folder_path = args.data_root
    rgb_images = os.listdir(os.path.join(data_folder_path, "rgb/"))
    depth_images = os.listdir(os.path.join(data_folder_path, "depth/"))

    if args.floor == 1:
        print("Start reconstructing the first floor...")
    if args.floor == 2:
        print("Start reconstructing the second floor...")
    reconstruct_start = time.time()
    print("Numbers of images is %d" % len(rgb_images))

    # temps
    pcd = []
    fpfh = []
    camera_pcd = []
    pcd_down = []
    pcd_transformed = [] # contain the pcd transformed to the main axis

    for index in range(1, len(rgb_images)):
    # Get the file path of RGB and depth images
        rgb_image_path = os.path.join(data_folder_path, "rgb/%d.png" % index)
        depth_image_path = os.path.join(data_folder_path, "depth/%d.png" % index)

        if index == 1:
            print("The principal picture is set as picture %d." % index)
            principal_pcd = depth_image_to_point_cloud(rgb_image_path, depth_image_path)
            pcd.append(principal_pcd[0])  # pcd[index-1]
            camera_pcd.append(principal_pcd[1])
            
            principal_pcd_down = preprocess_point_cloud(pcd[index - 1], voxel_size)
            pcd_down.append(principal_pcd_down[0])  # pcd_down[index-1]
            fpfh.append(principal_pcd_down[1])  # fpfh[index-1]
            pcd_transformed.append(pcd_down[index - 1])
        else:
            print("Processing picture %d..." % index)
            source_pcd = depth_image_to_point_cloud(rgb_image_path, depth_image_path)
            pcd.append(source_pcd[0])  # pcd[index-1]
            camera_pcd.append(source_pcd[1])
            
            source_pcd_down = preprocess_point_cloud(pcd[index - 1], voxel_size)
            pcd_down.append(source_pcd_down[0])  # pcd_down[index-1]
            fpfh.append(source_pcd_down[1])  # target_fpfh[index-1]
            
            # Perform global registration
            global_registration_start_time = time.time()
            global_registration_result = execute_global_registration(pcd_down[index - 1], pcd_transformed[index - 2], 
                                                                    fpfh[index - 1], fpfh[index - 2], voxel_size)
            print("Global registration took %.3f seconds." % (time.time() - global_registration_start_time))
        
            #ICP
            icp_start_time = time.time()
            if args.version == 'open3d':
                icp_result = local_icp_algorithm(pcd_down[index - 1], pcd_transformed[index - 2], 
                                                global_registration_result.transformation, voxel_size)
                transformation_matrix = icp_result.transformation  # transformation of index to index-1
            elif args.version == 'my_icp':
                icp_result = my_icp(pcd_down[index - 1], pcd_transformed[index - 2], 
                                    global_registration_result.transformation)
                transformation_matrix = icp_result  # transformation of index to index-1
                # draw_registration_result(pcd_down[index - 1], pcd_transformed[index - 2], transformation_matrix)
            print("ICP took %.3f seconds.\n" % (time.time() - icp_start_time))

            #Convert the point cloud to the coordinate system of the first camera
            pcd_transformed.append(pcd_down[index - 1].transform(transformation_matrix))
            camera_pcd[index - 1] = camera_pcd[index - 1].transform(transformation_matrix)
            
    # Merge the transformed point clouds
    point_cloud = accumulate_point_clouds(point_cloud, pcd_transformed)

    # Merge the camera point clouds
    estimate_camera_cloud = accumulate_point_clouds(estimate_camera_cloud, camera_pcd)

    estimate_camera_cloud.colors[0] = [0,0,0]
    estimate_lines = []
    for i in range(len(estimate_camera_cloud.points) - 1):
        estimate_lines.append([i, i + 1])
    estimate_line_set = o3d.geometry.LineSet()
    estimate_line_set.points = o3d.utility.Vector3dVector(estimate_camera_cloud.points)
    estimate_line_set.lines = o3d.utility.Vector2iVector(estimate_lines)
    estimate_line_set.paint_uniform_color([1, 0, 0])

    # filter the ceiling
    xyz_points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    if args.floor == 1:
        threshold_y = 0.01
    elif args.floor == 2:
        if args.version == 'open3d':
            threshold_y = 0.0115
        elif args.version == 'my_icp':
            threshold_y = 0.009
    filtered_xyz_points = xyz_points[xyz_points[:, 1] <= threshold_y]
    filtered_colors = colors[xyz_points[:, 1] <= threshold_y]
    point_cloud.points = o3d.utility.Vector3dVector(filtered_xyz_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    gt_pose_cloud, gt_line_set = load_pose_data(args)
    print("3D reconstruction took %.3f sec." % (time.time() - reconstruct_start))
    return point_cloud, gt_pose_cloud, gt_line_set, estimate_camera_cloud, estimate_line_set

def calculate_mean_l2_distance(gt_pos_pcd, estimate_camera_cloud):
    total_distance = 0
    num_points = len(estimate_camera_cloud.points)

    # Iterate over each point and calculate the distance
    for index in range(num_points):
        # Extract the respective coordinates
        gt_x = gt_pos_pcd.points[index][0]
        gt_y = gt_pos_pcd.points[index][1]
        gt_z = gt_pos_pcd.points[index][2]

        est_x = estimate_camera_cloud.points[index][0]
        est_y = estimate_camera_cloud.points[index][1]
        est_z = estimate_camera_cloud.points[index][2]

        # Compute the coordinate differences
        delta_x = gt_x - est_x
        delta_y = gt_y - est_y
        delta_z = gt_z - est_z

        # Calculate the Euclidean distance
        distance = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5
        total_distance += distance

    # Return the average distance
    average_distance = total_distance / num_points if num_points > 0 else 0
    return average_distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    result_pcd, gt_pos_pcd, line_set, estimate_camera_cloud, estimate_line_set= reconstruct(args)

    # TODO: Calculate and print L2 distance
    print("L2 distance:", calculate_mean_l2_distance(gt_pos_pcd, estimate_camera_cloud))

    # TODO: Visualize result
    o3d.visualization.draw_geometries([result_pcd,gt_pos_pcd,line_set,estimate_camera_cloud, estimate_line_set])