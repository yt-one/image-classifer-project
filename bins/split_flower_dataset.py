import os
import shutil

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

def move_img(imgs, root_dir, setname):
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for idx, path_img in enumerate(imgs):
        print(f"{idx}/{len(imgs)}")
        shutil.copy(path_img, data_dir)
    print(f"{setname} dataset, copy {len(imgs)} images to {data_dir}")


if __name__ == '__main__':
    import random

    # 0. config
    random_seed = 20251029
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    root_dir = ...

    # 1. 读取list, shuffle
    data_dir = os.path.join(root_dir, "jpg")
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith(".jpg")]
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs]
    random.seed(random_seed)
    random.shuffle(path_imgs)
    print(path_imgs[0])

    # 2. 划分
    train_breakpoints = int(len(path_imgs) * train_ratio)
    valid_breakpoints = int(len(path_imgs) * valid_ratio) + train_breakpoints

    train_imgs = path_imgs[:train_breakpoints]
    valid_imgs = path_imgs[train_breakpoints:valid_breakpoints]
    test_imgs = path_imgs[valid_breakpoints:]

    # 3. 复制，保存指定文件夹
    move_img(train_imgs, root_dir, "train")
    move_img(valid_imgs, root_dir, "valid")
    move_img(test_imgs, root_dir, "test")