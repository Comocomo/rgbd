import json
from pathlib import Path
from copy import deepcopy
from rgbd.configs import config_default

def load_default_config(key=None):

    cfg = deepcopy(config_default.cfg)

    if key is not None:
        assert key in cfg, '{} is not a valid config key'.format(key)
        cfg = cfg[key]

    return cfg

def merge_config(cfg_ref, cfg_update=None):
    return merge_dict_recursively(cfg_ref, cfg_update)

def merge_dict_recursively(ref, to_merge=None):

    result = deepcopy(ref)

    if to_merge is None:
        return result

    for key, val in to_merge.items():
        assert key in ref, '{} is not a valid key'.format(key)
        if key in result and isinstance(result[key], dict):
                result[key] = merge_dict_recursively(result[key], val)
        else:
            result[key] = deepcopy(val)

    return result

def convert_path_to_str_recusively(cfg):

    cfg_out = deepcopy(cfg)

    for key, val in cfg_out.items():
        if isinstance(cfg_out[key], dict):
                cfg_out[key] = convert_path_to_str_recusively(cfg_out[key])
        elif isinstance(val, Path):
            cfg_out[key] = val.as_posix()

    return cfg_out


def write_config(cfg, file_path):

    cfg_save = convert_path_to_str_recusively(cfg)

    cfg_json = json.dumps(cfg_save, indent=4, sort_keys=True)

    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    with open(file_path.resolve(), 'w+') as f:
        f.write(cfg_json)

    pass

def read_config(file_path):

    path = Path(file_path)
    text = path.read_text()
    cfg = json.loads(text)

    return cfg


    pass
