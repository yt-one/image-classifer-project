import os

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


# 这个类一般要自己写，因为需要根据实际数据（路径，文件名）调整
class FlowerDataset(Dataset):
    cls_num = 102
    names = tuple(range(cls_num))

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir:  数据根路径
        :param transform:  预留给预处理
        """

        self.root_dir = root_dir
        self.transform = transform

        self.img_info = [] # [(path, label), ..., ]
        self.label_array = None
        self._get_img_info()


    def __getitem__(self, index):  # 按下标取样， 可索引
        """
        输入标量index，从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """

        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label, path_img  # 返回多了图片路径 （bad case分析）

    def __len__(self): # 知道长度
        if len(self.img_info) == 0:
            raise Exception(f"\ndata dir{self.root_dir} is empty! Please check your dataset!")
        return len(self.img_info)

    def _get_img_info(self):
        # 读取硬盘里的数据 + 标签，存到list里 供__getitem__去使用

        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith(".jpg")]

        # 读取mat形式label
        label_file = "imagelabels"
        path_label_file = os.path.join(self.root_dir, '..', label_file)
        label_array = loadmat(path_label_file)['labels'].squeeze()
        self.label_array = label_array

        # 匹配label
        idx_imgs = [int(n[6:11]) for n in names_imgs]


        path_imgs = [os.path.join(self.root_dir, n) for n in names_imgs]

        # 注意索引，注意标签统一
        self.img_info = [(p, int(label_array[idx-1]-1))  for p, idx in zip(path_imgs, idx_imgs)]

# todo: 写完一个模块后需要单独测试