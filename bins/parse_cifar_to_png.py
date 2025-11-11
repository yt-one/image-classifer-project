import os
import pickle
import sys
import numpy as np
from cv2 import imwrite


def unpickle(file):
    """
    读取CIFAR-10的pickle文件
    """
    with open(file, 'rb') as fo:
        if sys.version_info < (3, 0):
            dict_ = pickle.load(fo)
        else:
            dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


def parse_pickle_img(pkl_data, index=0):
    """
    解析单个图像数据
    """
    # 获取指定索引的图像数据和标签
    img_data = pkl_data[b'data'][index]
    label = pkl_data[b'labels'][index]

    # 重塑图像数据 (3, 32, 32)
    img = np.reshape(img_data, (3, 32, 32))
    # 转换维度顺序 (chw -> hwc)
    img = img.transpose((1, 2, 0))

    return img, label


def check_data_dir(path_data):
    """
    检查数据目录是否存在
    """
    if not os.path.exists(path_data):
        print(f"{path_data} 数据目录不存在，请检查数据是否在指定路径...")
        return False
    return True


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cifar_dir = r"D:\Learn\Datasets\cifar-10-python"
    data_dir = os.path.join(cifar_dir, 'cifar-10-batches-py')
    if check_data_dir(data_dir):
        train_o_dir = os.path.join(cifar_dir, 'cifar10_train')
        test_o_dir = os.path.join(cifar_dir, 'cifar10_test')

        # train data
        for j in range(1,6):
            data_path = os.path.join(data_dir, f"data_batch_{j}")
            train_data = unpickle(data_path)
            print(data_path + " is loading...")

            for i in range(0, 10000):
                # 解析图片及标签
                img, label_num = parse_pickle_img(train_data,i)
                # 创建文件夹
                o_dir= os.path.join(train_o_dir, str(label_num))
                my_mkdir(o_dir)

                # 保存图片
                img_name = str(label_num) + '_' + str(i + (j - 1)*10000) + '.png'
                img_path = os.path.join(o_dir, img_name)

                imwrite(img_path, img)
            print(data_path + " loaded")

        # test data
        test_data_path = os.path.join(data_dir, "test_batch")
        test_data = unpickle(test_data_path)
        for i in range(0, 10000):
            # 解析图片及标签
            img, label_num = parse_pickle_img(test_data, i)

            # 创建类别文件夹
            o_dir = os.path.join(test_o_dir,str(label_num))
            my_mkdir(o_dir)

            # 保存图片
            img_name = str(label_num) + '__' + str(i) + '.png'
            img_path = os.path.join(o_dir, img_name)
            imwrite(img_path, img)

        print("done.")

