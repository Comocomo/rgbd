from pathlib import Path
from os import listdir
from os.path import exists, isfile, join, splitext
import re
import shutil

import open3d as o3d


def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=config["depth_scale"],
        depth_trunc=config["depth_max"],
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)

def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list

def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
            return path
    raise FileNotFoundError(
        f"None of the folders {folder_names} found in {path_dataset}")

def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth

def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_clean_folder(path_folder):

    path_folder = Path(path_folder)

    if path_folder.is_dir():
        shutil.rmtree(path_folder)
        path_folder.mkdir(parents=True)
    else:
        path_folder.mkdir(parents=True)

    pass