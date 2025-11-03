import os
import pickle
import shutil


def load_pickle(path_file):
    with open(path_file, 'rb') as f:
        data = pickle.load(f)
    return data

def my_mkdir(my_dir):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)

if __name__ == '__main__':
    path_pkl = ...
    data_root_dir = ...
    out_dir = ... # 输出文件目录
    error_info = load_pickle(path_pkl)

    for setname, info in error_info.items():
        for imgs_data in info:
            label, pred, path_img_rel = imgs_data
            path_img = os.path.join(data_root_dir, os.path.basename(path_img_rel))
            img_dir = os.path.join(out_dir, setname, str(label), str(pred)) # 图片文件夹
            my_mkdir(img_dir)
            shutil.copy(path_img, img_dir)