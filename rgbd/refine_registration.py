import os
import numpy as np
import open3d as o3d
import multiprocessing
from pathlib import Path

from rgbd.register_fragments import multiscale_icp, matching_result
from rgbd.optimize_posegraph import optimize_posegraph_for_refined_scene
from rgbd import utils


def update_posegraph_for_scene(s, t, transformation, information, odometry, pose_graph):

    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, transformation, information, uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, transformation, information, uncertain=True))

    return (odometry, pose_graph)


def local_refinement(source, target, transformation_init, cfg):

    voxel_size = cfg['voxel_size']
    voxel_sizes = [voxel_size / factor for factor in cfg['multiscale_icp_voxel_size_factors']]
    max_iterations = cfg['multiscale_icp_iterations']

    (transformation, information) = multiscale_icp(source, target, voxel_sizes, max_iterations, cfg, transformation_init)

    return (transformation, information)

def register_point_cloud_pair(ply_file_names, s, t, transformation_init, cfg):

    # print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    # print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])

    (transformation, information) = local_refinement(source, target, transformation_init, cfg)

    if cfg['debug_mode']:
        print(transformation)
        print(information)

    return (transformation, information)


def make_posegraph_for_refined_scene(ply_file_names, output_dir, cfg):

    pose_graph_path = output_dir / 'global_registration_optimized.json'
    pose_graph = o3d.io.read_pose_graph(pose_graph_path.as_posix())

    n_files = len(ply_file_names)
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id
        matching_results[s * n_files + t] = matching_result(s, t, edge.transformation)

    if cfg["python_multi_threading"] is True:
        os.environ['OMP_NUM_THREADS'] = '1'
        max_workers = max(1, min(multiprocessing.cpu_count() - 1, len(pose_graph.edges)))
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(processes=max_workers) as pool:
            args = [(ply_file_names, v.s, v.t, v.transformation, cfg) for k, v in matching_results.items()]
            results = pool.starmap(register_point_cloud_pair, args)

        for i, r in enumerate(matching_results):
            matching_results[r].transformation = results[i][0]
            matching_results[r].information = results[i][1]
    else:
        for r in matching_results:
            (matching_results[r].transformation, matching_results[r].information) = \
                register_point_cloud_pair(ply_file_names, matching_results[r].s, matching_results[r].t, matching_results[r].transformation, cfg)

    pose_graph_new = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph_new.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    for r in matching_results:
        (odometry, pose_graph_new) = update_posegraph_for_scene(matching_results[r].s, matching_results[r].t,
                                                                matching_results[r].transformation, matching_results[r].information,
                                                                odometry, pose_graph_new)

    pose_graph_path = output_dir / 'refined_registration_optimized.json'
    o3d.io.write_pose_graph(pose_graph_path.as_posix(), pose_graph_new)

    pass

def save_trajectory(output_dir):

    poses = []
    pose_graph_optimized_name = output_dir / 'global_registration_optimized.json'
    pose_graph_global = o3d.io.read_pose_graph(pose_graph_optimized_name.as_posix())

    fragment_id_file = 0  # FIXME: hardcoded, fix in in the future

    for fragment_id in range(len(pose_graph_global.nodes)):

        fragment_dir = 'fragments_1' if fragment_id == 0 else 'fragments_2'  # FIXME: hardcoded, fix in in the future
        pose_graph_fragment_name = output_dir.parent / fragment_dir / f'fragment_optimized_{fragment_id_file:03}.json'
        pose_graph_fragment = o3d.io.read_pose_graph(pose_graph_fragment_name.as_posix())

        for frame_id in range(len(pose_graph_fragment.nodes)):
            pose = np.dot(pose_graph_global.nodes[fragment_id].pose, pose_graph_fragment.nodes[frame_id].pose)
            poses.append(pose)
            pass

        pass

    traj_name = output_dir / 'trajectory.log'
    utils.write_poses_to_log(traj_name, poses)

    pass


def refine_registration_two_cameras(ply_file_names, output_dir, cfg):

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    make_posegraph_for_refined_scene(ply_file_names, output_dir, cfg)
    optimize_posegraph_for_refined_scene(output_dir, cfg)
    save_trajectory(output_dir)

    pass
