import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import datetime

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

    cfg_json = json.dumps(cfg_save, default=convert_array_to_json, indent=4, sort_keys=True)

    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    with open(file_path.resolve(), 'w+') as f:
        f.write(cfg_json)

    pass

def read_config(file_path):

    path = Path(file_path)
    text = path.read_text()
    cfg = json.loads(text, object_hook=convert_json_to_array)

    return cfg


    pass

def convert_array_to_json(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    if isinstance(x, datetime.date):
        return {"$date": x.isoformat()}
    raise TypeError(x)

def convert_json_to_array(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
        if key == "$date":
            return datetime.datetime.strptime(value, '%Y-%m-%d').date()
    return x
