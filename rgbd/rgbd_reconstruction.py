from pathlib import Path

from rgbd import config, utils
from rgbd.make_fragments import make_fragment_single_camera



class RgbdReconstruction:

    def __init__(self, path_dataset_1, path_dataset_2=None, output_root=None, cfg_update=None):

        if output_root is None:
            output_root = Path(path_dataset_1) / 'reconstruction'
        output_root = Path(output_root)

        cfg_update['path_dataset_1'] = Path(path_dataset_1)
        cfg_update['path_dataset_2'] = Path(path_dataset_2)
        cfg_update['output_root'] = output_root

            # update config
        cfg_default = config.load_default_config()
        self.cfg = config.merge_config(cfg_default, cfg_update)

        utils.make_clean_folder(output_root)

        config.write_config(self.cfg, output_root / 'config.json')

        pass

    def reconstruction_two_cameras(self):

        print('making fragments ...')
        make_fragment_single_camera(path_dataset=self.cfg['path_dataset_1'], output_dir=self.cfg['output_root'] / 'fragments_1', cfg=self.cfg['make_fragments'])
        make_fragment_single_camera(path_dataset=self.cfg['path_dataset_2'], output_dir=self.cfg['output_root'] / 'fragments_2', cfg=self.cfg['make_fragments'])
        print('done!')

        pass
