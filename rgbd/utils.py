from pathlib import Path
from os import listdir
from os.path import exists, isfile, join, splitext
import re
import shutil
import numpy as np
import open3d as o3d
import copy


def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, depth_scale, depth_max):

    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale, depth_max, convert_rgb_to_intensity)

    return rgbd_image

def filter_rgbd_by_depth(rgbd_image, depth_min_max=[1., 2.]):

    d = np.asarray(rgbd_image.depth)
    ind_filter = (d <= depth_min_max[0]) | (d >= depth_min_max[1])
    d[ind_filter] = 0
    rgbd_image.depth = o3d.geometry.Image(d)

    return rgbd_image

def filter_pcd(pcd_in, x_min_max=[-1., 1.], y_min_max=[-1., 1.], z_min_max=[0., 1.5], outlier_removal_flag=True, display=False):

    try:  # point cloud
        points = np.asarray(pcd_in.points)
    except:  # triangular mesh
        points = np.asarray(pcd_in.vertices)
        outlier_removal_flag = False  # no outlier removal function for meshes

    # filter by coordinate threshold
    ind = np.where((points[:, 0] > x_min_max[0]) & (points[:, 0] < x_min_max[1]) &
                   (points[:, 1] > y_min_max[0]) & (points[:, 1] < y_min_max[1]) &
                   (points[:, 2] > z_min_max[0]) & (points[:, 2] < z_min_max[1]))[0]

    pcd_filtered = copy.deepcopy(pcd_in)
    pcd_filtered = pcd_filtered.select_by_index(ind)

    #
    if outlier_removal_flag:
        dist_mean = 0.0012  # = calc_points_mean_dist(points, n_neighbors=5)  # e.g. 0.0012
        pcd_filtered = outlier_removal(pcd_filtered, dist_mean=dist_mean, radius_factor=20, nb_points=1000, iterations=1, display=False)

    if display:
        # change color of pcd_in
        color = [255. / 255, 140. / 255, 0. / 255]  # orange
        pcd_in.paint_uniform_color(color)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd filtered', width=1600, height=1400)
        vis.add_geometry(pcd_in)
        vis.add_geometry(pcd_filtered)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True

        vis.run()
        vis.destroy_window()

    return pcd_filtered


def outlier_removal(pcd_in, dist_mean=0.0012, radius_factor=15, nb_points=500, iterations=3, display=False):

    radius = radius_factor * dist_mean
    pcd_filtered = copy.deepcopy(pcd_in)
    ind_list = []
    for n in range(iterations):
        cl, ind = pcd_filtered.remove_radius_outlier(nb_points, radius)
        ind_list.append(ind)  # FIXME: need to treat different number of points in each iteration
        if display:
            display_inlier_outlier(pcd_filtered, ind)
        pcd_filtered = pcd_filtered.select_by_index(ind)

    return pcd_filtered


def mesh_largest_connected_component(mesh, display=False, save_file_name=None):

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)

    if display:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # save filtered mesh
    if save_file_name is not None:
        o3d.io.write_triangle_mesh(save_file_name, mesh, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

    return mesh


def display_inlier_outlier(cloud, ind):
    """
    source: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html#Select-down-sample
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024],
                                      )
    pass



def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)

def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list

def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
            return path
    raise FileNotFoundError(
        f"None of the folders {folder_names} found in {path_dataset}")

def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth

def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_clean_folder(path_folder):

    path_folder = Path(path_folder)

    if path_folder.is_dir():
        shutil.rmtree(path_folder)
        path_folder.mkdir(parents=True)
    else:
        path_folder.mkdir(parents=True)

    pass