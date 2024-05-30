from pathlib import Path
import numpy as np
import open3d as o3d
from pytest import approx

from rgbd import RgbdReconstruction, utils

np.set_printoptions(suppress=True)


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
        utils.draw_geometries_wrapper(fragment_1_pcd, window_name='fragment_1_pcd')

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
        utils.draw_geometries_wrapper(fragment_1_pcd, window_name='fragment_1_pcd')
        utils.draw_geometries_wrapper(fragment_2_pcd, window_name='fragment_2_pcd')


    assert np.asarray(fragment_1_pcd.points).shape == (264969, 3)
    assert np.asarray(fragment_2_pcd.points).shape == (243206, 3)

    pass


def test_register_fragments_two_cameras():

    output_root = Path(__file__).parent / 'output' / 'rgbd_reconstruction_two_cameras'

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

    # optimized pose graph
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


def test_refine_registration():

    output_root = Path(__file__).parent / 'output' / 'rgbd_reconstruction_two_cameras'

    cfg_update = {}

    rec = RgbdReconstruction(output_root=output_root, cfg_update=cfg_update)

    rec.refine_registration()

    # refined pose graph
    pose_graph_path = output_root / 'scene' / 'refined_registration_optimized.json'
    pose_graph = o3d.io.read_pose_graph(pose_graph_path.as_posix())

    assert len(pose_graph.edges) == 1
    assert pose_graph.edges[0].transformation == approx(np.array([[ 0.58451919,  0.43685067, -0.68373885,  0.87620105],
                                                                  [-0.41363186,  0.88539746,  0.21208494, -0.26000005],
                                                                  [ 0.69803008,  0.15884845,  0.69822717,  0.36592881],
                                                                  [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                                        abs=1e-4)
    assert len(pose_graph.nodes) == 2
    assert pose_graph.nodes[0].pose == approx(np.array([[1., 0., 0., 0.],
                                                        [0., 1., 0., 0.],
                                                        [0., 0., 1., 0.],
                                                        [0., 0., 0., 1.]]),
                                              abs=1e-4)
    assert pose_graph.nodes[1].pose == approx(np.array([[ 0.58451919, -0.41363187,  0.69803008, -0.87512995],
                                                        [ 0.43685067,  0.88539746,  0.15884845, -0.21069286],
                                                        [-0.68373884,  0.21208494,  0.69822716,  0.39873334],
                                                        [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                              abs=1e-4)
    # trajectory file
    trajectory_path = output_root / 'scene' / 'trajectory.log'
    trajectory = utils.read_poses_from_log(trajectory_path)

    assert len(trajectory) == 4
    assert trajectory[0] == approx(np.array([[1., 0., 0., 0.],
                                             [0., 1., 0., 0.],
                                             [0., 0., 1., 0.],
                                             [0., 0., 0., 1.]]),
                                   abs=1e-4)
    assert trajectory[1] == approx(np.array([[ 0.99999775, -0.00076531, -0.00198032,  0.00306604],
                                             [ 0.00076188,  0.99999821, -0.0017302 ,  0.00295231],
                                             [ 0.00198164,  0.00172869,  0.99999654, -0.00228152],
                                             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                   abs=1e-4)
    assert trajectory[2] == approx(np.array([[ 0.58985467, -0.41571865,  0.69227846, -0.86909786],
                                             [ 0.44180479,  0.88374917,  0.15425929, -0.20597051],
                                             [-0.67592898,  0.21486138,  0.70495007,  0.38841977],
                                             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                   abs=1e-4)
    assert trajectory[3] == approx(np.array([[ 0.59014578, -0.4163221 ,  0.69166745, -0.86888836],
                                             [ 0.44276768,  0.88332877,  0.15390604, -0.20548039],
                                             [-0.67504424,  0.21542099,  0.70562671,  0.38755961],
                                             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                                   abs=1e-4)

    pass



if __name__ == '__main__':

    # test_make_fragments_single_camera()
    # test_make_fragments_two_cameras()
    # test_register_fragments_two_cameras()
    test_refine_registration()

    pass