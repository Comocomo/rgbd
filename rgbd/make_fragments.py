import open3d as o3d
import numpy as np

from rgbd.utils import get_rgbd_file_lists, read_rgbd_image
from rgbd.opencv_pose_estimation import pose_estimation


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config):

    convert_rgb_to_intensity = config['convert_rgb_to_intensity']
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], convert_rgb_to_intensity, config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], convert_rgb_to_intensity, config)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["depth_diff_max"]
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

                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments - 1, s, t))

                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, cfg)

                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                                                 t - sid,
                                                                                 trans,
                                                                                 info,
                                                                                 uncertain=False))

            # keyframe loop closure
            if (s % cfg['n_keyframes_per_n_frame'] == 0 and t % cfg['n_keyframes_per_n_frame'] == 0) \
                    or (((s < sid_2) and (t >= sid_2)) or ((s >= sid_2) and (t < sid_2))):
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (
                fragment_id, n_fragments - 1, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                                                                with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(s - sid, t - sid, trans, info, uncertain=True))

    pose_graph_file = cfg['output_fragment'] / f'', config["template_fragment_posegraph"] % fragment_id)
    Path(pose_graph_file).parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_pose_graph(pose_graph_file, pose_graph)

    pass


def make_fragment_single_camera(path_dataset, output_dir, cfg):

    path_intrinsics = path_dataset / 'intrinsics.json'
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(path_intrinsics)

    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)

    n_max_images = cfg['make_fragments']['n_max_images']

    fragment_id = 0
    n_fragments = 1

    sid = 0
    eid = sid + n_max_images

    color_files = color_files[sid:eid]
    depth_files = depth_files[sid:eid]

    make_posegraph_for_fragment_two_cameras(color_files, depth_files,
                                            sid_1, eid_1, sid_2, eid_2,
                                            fragment_id, n_fragments,
                                            intrinsic_1, intrinsic_2,
                                            cfg, with_opencv=True,
                                            trans_init_1=np.identity(4),
                                            trans_init_2=T_init)
    # optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, cfg)
    # make_pointcloud_for_fragment(config["path_dataset"], color_files, depth_files, fragment_id, n_fragments, intrinsic_1, cfg, intrinsic_2, sid_2)


    pass

