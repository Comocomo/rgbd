from pathlib import Path
import numpy as np
import open3d as o3d
from pytest import approx

from rgbd import RgbdReconstruction


def test_make_fragments_single_camera():

    dataset_1_path = (Path(__file__).parent / 'test_data' / '20240506_175654_IQ_left').as_posix()
    output_root = (Path(__file__).parent / 'output' / 'rgbd_reconstruction_single_camera').as_posix()

    n_max_images = 2  # default is None

    cfg_update = {'make_fragments': {'n_max_images': n_max_images},
                                     }
    rec = RgbdReconstruction(dataset_1_path, output_root=output_root, cfg_update=cfg_update)

    rec.make_fragments()

    fragment_1_pcd = o3d.io.read_point_cloud(rec.data['fragment_1_pcd_path'].as_posix())

    display = False
    if display:
        o3d.visualization.draw_geometries([fragment_1_pcd], window_name='fragment_1_pcd', mesh_show_back_face=True)

    assert np.asarray(fragment_1_pcd.points).shape == (264969, 3)
    assert np.asarray(rec.data['fragment_2_pcd_path'] is None)

    pass



def test_make_fragments_two_cameras():

    dataset_1_path = (Path(__file__).parent / 'test_data' / '20240506_175654_IQ_left').as_posix()
    dataset_2_path = (Path(__file__).parent / 'test_data' / '20240506_175527_IQ_right').as_posix()
    output_root = (Path(__file__).parent / 'output' / 'rgbd_reconstruction_two_cameras').as_posix()

    n_max_images = 2  # default is None

    cfg_update = {'make_fragments': {'n_max_images': n_max_images},
                                     }
    rec = RgbdReconstruction(dataset_1_path, dataset_2_path, output_root, cfg_update)

    rec.make_fragments()

    fragment_1_pcd = o3d.io.read_point_cloud(rec.data['fragment_1_pcd_path'].as_posix())
    fragment_2_pcd = o3d.io.read_point_cloud(rec.data['fragment_2_pcd_path'].as_posix())

    display = False
    if display:
        o3d.visualization.draw_geometries([fragment_1_pcd], window_name='fragment_1_pcd', mesh_show_back_face=True)
        o3d.visualization.draw_geometries([fragment_2_pcd], window_name='fragment_2_pcd', mesh_show_back_face=True)


    assert np.asarray(fragment_1_pcd.points).shape == (264969, 3)
    assert np.asarray(fragment_2_pcd.points).shape == (243206, 3)

    pass


def test_register_fragments_two_cameras():

    output_root = Path(__file__).parent / 'output' / 'rgbd_reconstruction_two_cameras'

    # assume that test_make_fragments_two_cameras() run before and saved following ply files
    fragment_1_pcd_path = output_root / 'fragments_1' / 'fragment_0.ply'
    fragment_2_pcd_path = output_root / 'fragments_2' / 'fragment_0.ply'

    # direct transformation saved in gitlab between right and left cameras of IQ 2.3 (f1150179-f0350845)
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957/1000],
                  [0.45228962,    0.87499056,    0.17270096, -242.94946348/1000],
                  [-0.64597401,   0.18787587,    0.73987852,  344.81816623/1000],
                  [-0,        -0,        -0,         1],
                  ])

    T_init_1to2 = np.linalg.inv(T)  # use inverse transformation

    cfg_update = {'register_fragments': {'T_init_1to2': T_init_1to2},
                  }

    rec = RgbdReconstruction(output_root=output_root, cfg_update=cfg_update)

    rec.register_fragments()

    # initial pose graph (before optimization)
    pose_graph_path = output_root / 'scene' / 'global_registration.json'
    pose_graph = o3d.io.read_pose_graph(pose_graph_path.as_posix())

    assert len(pose_graph.edges) == 1
    assert pose_graph.edges[0].transformation == approx(np.array([[ 0.58985467,  0.4418048 , -0.67592898,  0.86618438],
                                                                  [-0.41571864,  0.88374917,  0.21486138, -0.26273033],
                                                                  [ 0.69227846,  0.15425929,  0.70495007,  0.35961405],
                                                                  [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                                        abs=1e-4)
    assert len(pose_graph.nodes) == 2
    assert pose_graph.nodes[0].pose == approx(np.array([[1., 0., 0., 0.],
                                                        [0., 1., 0., 0.],
                                                        [0., 0., 1., 0.],
                                                        [0., 0., 0., 1.]]),
                                              abs=1e-4)
    assert pose_graph.nodes[1].pose == approx(np.array([[ 0.58985467, -0.41571865,  0.69227846, -0.86909786],
                                                        [ 0.44180479,  0.88374917,  0.15425929, -0.20597051],
                                                        [-0.67592898,  0.21486138,  0.70495007,  0.38841977],
                                                        [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                              abs=1e-4)

    # initial pose graph (before optimization)
    pose_graph_optimized_path = output_root / 'scene' / 'global_registration_optimized.json'
    pose_graph_optimized = o3d.io.read_pose_graph(pose_graph_optimized_path.as_posix())

    assert len(pose_graph_optimized.edges) == 1
    assert pose_graph_optimized.edges[0].transformation == approx(np.array([[ 0.58985467,  0.4418048 , -0.67592898,  0.86618438],
                                                                            [-0.41571864,  0.88374917,  0.21486138, -0.26273033],
                                                                            [ 0.69227846,  0.15425929,  0.70495007,  0.35961405],
                                                                            [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                                                  abs=1e-4)
    assert len(pose_graph_optimized.nodes) == 2
    assert pose_graph_optimized.nodes[0].pose == approx(np.array([[1., 0., 0., 0.],
                                                                  [0., 1., 0., 0.],
                                                                  [0., 0., 1., 0.],
                                                                  [0., 0., 0., 1.]]),
                                                        abs=1e-4)
    assert pose_graph_optimized.nodes[1].pose == approx(np.array([[ 0.58985467, -0.41571865,  0.69227846, -0.86909786],
                                                                  [ 0.44180479,  0.88374917,  0.15425929, -0.20597051],
                                                                  [-0.67592898,  0.21486138,  0.70495007,  0.38841977],
                                                                  [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                                        abs=1e-4)



    pass


if __name__ == '__main__':

    # test_make_fragments_single_camera()
    # test_make_fragments_two_cameras()
    test_register_fragments_two_cameras()

    pass