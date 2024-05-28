from pathlib import Path

import open3d as o3d
from os.path import join

def run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
                               max_correspondence_distance,
                               preference_loop_closure):

    # to display messages from o3d.pipelines.registration.global_optimization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # set optimization parameters
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=max_correspondence_distance,
                                                                 edge_prune_threshold=0.25,
                                                                 preference_loop_closure=preference_loop_closure,
                                                                 reference_node=0)

    # optimize
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
    o3d.io.write_pose_graph(pose_graph_optimized_name, pose_graph)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    pass


def optimize_posegraph_for_fragment(posegraph_dir, fragment_id, cfg):

    pose_graph_name = posegraph_dir / f'fragment_{fragment_id}.json'
    pose_graph_optimized_name = posegraph_dir / f'fragment_optimized_{fragment_id}.json'

    run_posegraph_optimization(pose_graph_name.as_posix(), pose_graph_optimized_name.as_posix(),
                               max_correspondence_distance=cfg["depth_diff_max"],
                               preference_loop_closure=cfg["preference_loop_closure_odometry"])

    pass

def optimize_posegraph_for_scene(posegraph_dir, cfg):

    pose_graph_name = posegraph_dir / 'global_registration.json'
    pose_graph_optimized_name = posegraph_dir / 'global_registration_optimized.json'

    run_posegraph_optimization(pose_graph_name.as_posix(), pose_graph_optimized_name.as_posix(),
                               max_correspondence_distance=cfg["voxel_size"] * 1.4,
                               preference_loop_closure=cfg["preference_loop_closure_registration"])
    pass

def optimize_posegraph_for_refined_scene(path_dataset, config):
    pose_graph_name = join(path_dataset, config["template_refined_posegraph"])
    pose_graph_optimized_name = join(path_dataset, config["template_refined_posegraph_optimized"])
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
                               max_correspondence_distance=config["voxel_size"] * 1.4,
                               preference_loop_closure=config["preference_loop_closure_registration"])
    pass