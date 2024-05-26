from pathlib import Path
import numpy as np

from rgbd import RgbdReconstruction


def test_make_fragments_single_camera():

    dataset_1_path = (Path(__file__).parent / 'test_data' / '20240506_175654_IQ_left').as_posix()
    output_root = (Path(__file__).parent / 'output' / 'rgbd_reconstruction_single_camera').as_posix()

    n_max_images = 2  # default is None

    cfg_update = {'make_fragments': {'n_max_images': n_max_images},
                                     }
    rec = RgbdReconstruction(dataset_1_path, output_root=output_root, cfg_update=cfg_update)

    rec.reconstruction_single_camera()

    assert np.asarray(rec.fragment_1_pcd.points).shape == (264969, 3)

    pass



def test_make_fragments_two_cameras():

    dataset_1_path = (Path(__file__).parent / 'test_data' / '20240506_175654_IQ_left').as_posix()
    dataset_2_path = (Path(__file__).parent / 'test_data' / '20240506_175527_IQ_right').as_posix()
    output_root = (Path(__file__).parent / 'output' / 'rgbd_reconstruction_two_cameras').as_posix()

    n_max_images = 2  # default is None

    cfg_update = {'make_fragments': {'n_max_images': n_max_images},
                                     }
    rec = RgbdReconstruction(dataset_1_path, dataset_2_path, output_root, cfg_update)

    rec.reconstruction_two_cameras()

    assert np.asarray(rec.fragment_1_pcd.points).shape == (264969, 3)
    assert np.asarray(rec.fragment_2_pcd.points).shape == (243206, 3)

    pass


# def test_register_fragments_two_cameras():
#
#     # direct transformation saved in gitlab between right and left cameras of IQ 2.3 (f1150179-f0350845)
#     # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
#     T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957],
#                   [0.45228962,    0.87499056,    0.17270096, -242.94946348],
#                   [-0.64597401,   0.18787587,    0.73987852,  344.81816623],
#                   [-0,        -0,        -0,         1],
#                   ])
#     T_init_2to1 = np.linalg.inv(T)  # use inverse transformation
#
#
#     pass


if __name__ == '__main__':

    test_make_fragments_single_camera()
    # test_make_fragments_two_cameras()

    pass