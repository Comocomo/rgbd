from pathlib import Path

from rgbd import config, utils
from rgbd.make_fragments import make_fragment_single_camera
from rgbd.register_fragments import register_fragments_two_cameras


class RgbdReconstruction:

    def __init__(self, path_dataset_1=None, path_dataset_2=None, output_root=None, cfg_update=None):

        if output_root is None:
            output_root = Path(path_dataset_1) / 'reconstruction'
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        cfg_update['path_dataset_1'] = Path(path_dataset_1) if path_dataset_1 is not None else None
        cfg_update['path_dataset_2'] = Path(path_dataset_2) if path_dataset_2 is not None else None
        cfg_update['output_root'] = output_root

            # update config
        cfg_default = config.load_default_config()
        self.cfg = config.merge_config(cfg_default, cfg_update)

        config.write_config(self.cfg, output_root / 'config.json')

        self.data = {'fragment_1_pcd_path': None,
                     'fragment_2_pcd_path': None,
                     }

        pass

    def reconstruction_two_cameras(self):

        self.make_fragments()
        self.register_fragments()
        # self.refine_registration()
        # self.integrate_scene()

        pass

    def make_fragments(self):

        print('making fragments ...')
        self.data['fragment_1_pcd_path'] = make_fragment_single_camera(path_dataset=self.cfg['path_dataset_1'], output_dir=self.cfg['output_root'] / 'fragments_1', cfg=self.cfg['make_fragments'])
        if self.cfg['path_dataset_2'] is not None:
            self.data['fragment_2_pcd_path'] = make_fragment_single_camera(path_dataset=self.cfg['path_dataset_2'], output_dir=self.cfg['output_root'] / 'fragments_2', cfg=self.cfg['make_fragments'])
        print('done!')


    def register_fragments(self):

        fragment_1_pcd_path = self.cfg['output_root'] / 'fragments_1' / 'fragment_0.ply'
        fragment_2_pcd_path = self.cfg['output_root'] / 'fragments_2' / 'fragment_0.ply'
        ply_file_names = [fragment_1_pcd_path.as_posix(), fragment_2_pcd_path.as_posix()]

        register_fragments_two_cameras(ply_file_names, self.cfg['output_root'] / 'scene', self.cfg['register_fragments'])

        pass
