import os
import numpy as np
import open3d as o3d
import multiprocessing
import copy

from rgbd.optimize_posegraph import optimize_posegraph_for_scene
from rgbd.utils import make_clean_folder


def preprocess_point_cloud(pcd, cfg):
    voxel_size = cfg["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
    return (pcd_down, pcd_fpfh)


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, cfg):

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    distance_threshold = cfg["voxel_size"] * 1.4

    if cfg["global_registration"] == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching( source, target, source_fpfh, target_fpfh,
                                                                                        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))

    if cfg["global_registration"] == "ransac":
        # Fallback to preset parameters that works better
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))

    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))

    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, distance_threshold, result.transformation)

    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)))

    return (True, result.transformation, information)

def compute_initial_registration(s, t, source_down, target_down, source_fpfh, target_fpfh, cfg, trans_init=None):

    if t == s + 1:  # odometry case
        print("Using RGBD odometry")
        if trans_init is None:
            # pose_graph_frag = o3d.io.read_pose_graph(join(path_dataset, config["template_fragment_posegraph_optimized"] % s))
            # n_nodes = len(pose_graph_frag.nodes)
            # transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes - 1].pose)
            pass
        else:
            transformation_init = trans_init

        (transformation, information) = multiscale_icp(source_down, target_down, [cfg["voxel_size"]], [50], cfg, transformation_init)

    else:  # loop closure case
        (success, transformation, information) = register_point_cloud_fpfh(source_down, target_down, source_fpfh, target_fpfh, cfg)

        if not success:
            print("No reasonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6, 6)))

    print(transformation)

    if cfg["debug_mode"]:
        # draw_registration_result(source_down, target_down, trans_init)  # debug
        draw_registration_result(source_down, target_down, transformation)

    return (True, transformation, information)

def update_posegraph_for_scene(s, t, transformation, information, odometry,
                               pose_graph):

    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, transformation, information, uncertain=False))

    else:  # loop closure case
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, transformation, information, uncertain=True))

    return (odometry, pose_graph)

def register_point_cloud_pair(ply_file_names, s, t, cfg, trans_init=None):

    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])

    (source_down, source_fpfh) = preprocess_point_cloud(source, cfg)
    (target_down, target_fpfh) = preprocess_point_cloud(target, cfg)

    (success, transformation, information) = compute_initial_registration(s, t, source_down, target_down, source_fpfh, target_fpfh, cfg, trans_init)

    if t != s + 1 and not success:
        return (False, np.identity(4), np.identity(6))

    if cfg["debug_mode"]:
        print(transformation)
        print(information)

    return (True, transformation, information)

def multiscale_icp(source, target, voxel_size, max_iter, cfg, init_transformation=np.identity(4)):

    current_transformation = init_transformation

    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = cfg["voxel_size"] * 1.4
        print("voxel_size {}".format(voxel_size[scale]))
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])

        if cfg["icp_method"] == "point_to_point":
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter))
        else:

            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0, max_nn=30))
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0, max_nn=30))

            if cfg["icp_method"] == "point_to_plane":
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter))

            if cfg["icp_method"] == "color":
                # Colored ICP is sensitive to threshold.
                # Fallback to preset distance threshold that works better.
                # TODO: make it adjustable in the upgraded system.
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, voxel_size[scale],
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))

            if cfg["icp_method"] == "generalized":
                result_icp = o3d.pipelines.registration.registration_generalized_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))

        current_transformation = result_icp.transformation

        if i == len(max_iter) - 1:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source_down, target_down, voxel_size[scale] * 1.4, result_icp.transformation)

    if cfg["debug_mode"]:
        # draw_registration_result_original_color(source, target, init_transformation)
        draw_registration_result_original_color(source, target, result_icp.transformation)

    return (result_icp.transformation, information_matrix)



def draw_registration_result(source, target, transformation):
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    pass

def draw_registration_result_original_color(source, target, transformation):
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    pass


class matching_result:

    def __init__(self, s, t, trans_init=np.identity(4)):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans_init
        self.information = np.identity(6)
        pass


def make_posegraph_for_scene(ply_file_names, output_dir, cfg, trans_init=np.identity(4)):

    pose_graph = o3d.pipelines.registration.PoseGraph()
    # odometry = trans_init
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t, trans_init)

    if cfg["python_multi_threading"] is True:
        os.environ['OMP_NUM_THREADS'] = '1'
        max_workers = max(1, min(multiprocessing.cpu_count() - 1, len(matching_results)))
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(processes=max_workers) as pool:
            args = [(ply_file_names, v.s, v.t, cfg) for k, v in matching_results.items()]
            results = pool.starmap(register_point_cloud_pair, args)

        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]

    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation, matching_results[r].information) = \
                register_point_cloud_pair(ply_file_names, matching_results[r].s, matching_results[r].t, cfg, trans_init)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegraph_for_scene(matching_results[r].s, matching_results[r].t,
                                                                matching_results[r].transformation,
                                                                matching_results[r].information, odometry, pose_graph)

    pose_graph_path = output_dir / 'global_registration.json'
    o3d.io.write_pose_graph(pose_graph_path.as_posix(), pose_graph)

    pass


def register_fragments_two_cameras(ply_file_names, output_dir, cfg):

    make_clean_folder(output_dir)
    make_posegraph_for_scene(ply_file_names, output_dir, cfg, cfg['T_init_1to2'])
    optimize_posegraph_for_scene(output_dir, cfg)

    pass