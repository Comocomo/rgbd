from pathlib import Path

from rgbd import config, utils
from rgbd.make_fragments import make_fragment_two_cameras



class RgbdReconstruction:

    def __init__(self, path_dataset_1, path_dataset_2=None, output_root=None, cfg_update=None):

        if output_root is None:
            output_root = Path(path_dataset_1) / 'reconstruction'
        output_root = Path(output_root)

        cfg_update['path_dataset_1'] = Path(path_dataset_1)
        cfg_update['path_dataset_2'] = Path(path_dataset_2)
        cfg_update['output_root'] = output_root
        cfg_update['output_fragments_1'] = output_root / 'fragments_1'  # camera 1 fragments
        cfg_update['output_fragments_2'] = output_root / 'fragments_2'  # camera 2 fragments
        cfg_update['output_scene'] = output_root / 'scene'

            # update config
        cfg_default = config.load_default_config()
        self.cfg = config.merge_config(cfg_default, cfg_update)

        utils.make_clean_folder(output_root)

        pass

    def reconstruction_two_cameras(self):

        make_fragment_two_cameras(self.cfg)

        pass
