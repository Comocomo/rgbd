from pathlib import Path
import open3d as o3d
import numpy as np

from rgbd.utils import get_rgbd_file_lists, read_rgbd_image, filter_rgbd_by_depth, filter_pcd, mesh_largest_connected_component
from rgbd.opencv_pose_estimation import pose_estimation
from rgbd.optimize_posegraph import optimize_posegraph_for_fragment


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, convert_rgb_to_intensity=True,
                           depth_diff_max=0.03, depth_scale=3999.999810010204, depth_max=2.5):

    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], convert_rgb_to_intensity, depth_scale, depth_max)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], convert_rgb_to_intensity, depth_scale, depth_max)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = depth_diff_max
    if abs(s - t) != 1:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image, target_rgbd_image, intrinsic, False)
            if success_5pt:
                [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(source_rgbd_image, target_rgbd_image, intrinsic, odo_init, o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def make_posegraph_for_fragment(color_files, depth_files,
                                sid, eid,
                                fragment_id, n_fragments,
                                intrinsic,
                                cfg, with_opencv=True,
                                trans_init=np.identity(4),
                                output_dir=None
                                ):

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pose_graph = o3d.pipelines.registration.PoseGraph()

    trans_odometry = trans_init

    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):

            # odometry
            if t == s + 1:

                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, cfg['depth_scale'], cfg['depth_max'])

                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                                                 t - sid,
                                                                                 trans,
                                                                                 info,
                                                                                 uncertain=False))

            # keyframe loop closure
            if (s % cfg['n_keyframes_per_n_frame'] == 0 and t % cfg['n_keyframes_per_n_frame'] == 0):
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, cfg['depth_scale'], cfg['depth_max'])
                if success:
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid, t - sid, trans, info, uncertain=True))

    pose_graph_file = output_dir / f'fragment_{fragment_id}.json'
    Path(pose_graph_file).parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_pose_graph(pose_graph_file.as_posix(), pose_graph)

    pass

def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id, n_fragments, pose_graph_name, intrinsic, cfg):

    pose_graph = o3d.io.read_pose_graph(pose_graph_name)

    # TSDF arguments: https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html#TSDF-volume-integration
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=cfg["tsdf_cubic_size"] / 512.0,  # 0.75 [m] / 512 = 1.46 [mm]
                                                          sdf_trunc=0.04,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(pose_graph.nodes)):

        i_abs = fragment_id * cfg['n_frames_per_fragment'] + i
        print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." % (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))

        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False, cfg['depth_scale'], cfg['depth_max'])
        rgbd = filter_rgbd_by_depth(rgbd, depth_min_max=cfg['z_min_max'])
        pose = pose_graph.nodes[i].pose

        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

        pass

    mesh = volume.extract_triangle_mesh()
    mesh = filter_pcd(mesh, cfg['x_min_max'], cfg['y_min_max'], cfg['z_min_max'], outlier_removal_flag=False, display=False)
    mesh = mesh_largest_connected_component(mesh, display=False, save_file_name=None)
    mesh.compute_vertex_normals()

    return mesh

def make_pointcloud_for_fragment(posegraph_dir, color_files, depth_files, fragment_id, n_fragments, intrinsic, cfg):

    pose_graph_optimized_name = posegraph_dir / f'fragment_optimized_{fragment_id}.json'
    mesh = integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id, n_fragments, pose_graph_optimized_name.as_posix(), intrinsic, cfg)

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors

    pcd_name = posegraph_dir / f'fragment_{fragment_id}.ply'
    o3d.io.write_point_cloud(pcd_name.as_posix(), pcd, write_ascii=False, compressed=True)

    return pcd


def make_fragment_single_camera(path_dataset, output_dir, cfg):

    path_intrinsics = path_dataset / 'intrinsics.json'
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(path_intrinsics.as_posix())

    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)

    n_max_images = cfg['n_max_images']

    fragment_id = 0
    n_fragments = 1

    sid = 0
    eid = sid + n_max_images

    color_files = color_files[sid:eid]
    depth_files = depth_files[sid:eid]

    make_posegraph_for_fragment(color_files, depth_files,
                                sid, eid,
                                fragment_id, n_fragments,
                                intrinsics,
                                cfg, with_opencv=True,
                                trans_init=np.identity(4),
                                output_dir=output_dir,
                                )
    optimize_posegraph_for_fragment(output_dir, fragment_id, cfg)
    pcd = make_pointcloud_for_fragment(output_dir, color_files, depth_files, fragment_id, n_fragments, intrinsics, cfg)

    return pcd

