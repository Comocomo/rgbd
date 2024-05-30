import numpy as np

paths = {'path_dataset_1': None,
         'path_dataset_2': None,
         'output_root': None,
         }

make_fragments = {'n_max_images': None,
                  'n_keyframes_per_n_frame': 1,
                  'n_frames_per_fragment': 30,
                  'depth_scale': 3999.999810010204,
                  'depth_max': 2.,
                  'depth_diff_max': 0.03,
                  'preference_loop_closure_odometry': 0.1,
                  'tsdf_cubic_size': 0.75,  # voxel size is 0.75 [m] / 512 = 1.46 [mm]
                  # mesh filtering thresholds
                  'x_min_max': [-1., 1.],
                  'y_min_max': [-1., 1.],
                  'z_min_max': [1., 2.],
                  'outlier_removal_flag': True
                  }

register_fragments = {'T_init_1to2': np.identity(4),
                      'voxel_size': 0.01,  # [m], used for downsampling pcd for initial registration
                      'icp_method': 'color',  # one of ['point_to_point', 'point_to_plane', 'color', 'generalized']
                      'preference_loop_closure_registration': 5.0,
                      'python_multi_threading': False,
                      'debug_mode': False,
                      }

refine_registration = {'voxel_size': 0.01,
                       'icp_method': 'color',  # one of ['point_to_point', 'point_to_plane', 'color', 'generalized']
                       'multiscale_icp_voxel_size_factors': [1., 2., 4.],  # divide voxel size by these factors
                       'multiscale_icp_iterations': [50, 30, 14],
                       'preference_loop_closure_registration': 5.0,
                       'python_multi_threading': False,
                       'debug_mode': False,
                       }


cfg = {'make_fragments': make_fragments,
       'register_fragments': register_fragments,
       'refine_registration': refine_registration,
       }

cfg.update(paths)