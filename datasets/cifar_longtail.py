import os
import random

from PIL import Image
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", 'horse', 'ship', 'truck')
    cls_num = len(names)

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # 定义list用于存储样本路径、标签

        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]

        # 打开图像并转换为RGB
        img = Image.open(path_img).convert('RGB')

        # 应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception(f"Dataset directory '{self.root_dir}' is empty! Please check your path to images!")
        return len(self.img_info)

    def _get_img_info(self):
        # 遍历根目录下的所有子目录
        for root, dirs, _ in os.walk(self.root_dir):
            # 遍历类别文件夹
            for sub_dir in dirs:
                # 获取该类别文件夹下的所有PNG图像文件
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.abspath(os.path.join(root, sub_dir, img_name))
                    label = int(sub_dir)  # 文件夹名称就是标签数字
                    self.img_info.append((path_img, int(label)))

            # 只处理第一级目录，避免递归子目录
            break

        # 将数据顺序打乱
        random.shuffle(self.img_info)

class CifarLTDataset(CifarDataset):
    def __init__(self, root_dir, transform=None, imb_factor=0.01, isTrain=True):
        """
        :param root_dir:
        :param transform:
        :param imb_type:
        :param imb_factor: float, 值越小，数量下降越快,0.1表示最少的类是最多的类(99.1倍，如360、3600
        :param isTrain:
        """
        super(CifarLTDataset, self).__init__(root_dir, transform=transform)
        self.imb_factor = imb_factor
        if isTrain:
            self.num_per_cls = self._get_img_num_per_cls()    # 计算每个类的样本数
            self._select_img()    # 采样获得符合长尾分布的数据量
        else:
            # 非训练状态，可采用均衡数据集测试
            self.num_per_cls = []
            for n in range(self.cls_num):
                label_list = [label for p, label in self.img_info]    # 获取每个标签
                self.num_per_cls.append(label_list.count(n))    # 统计每个类别数量

    def _select_img(self):
        new_lst = []
        for n, img_num in enumerate(self.num_per_cls):
            lst_tmp = [info for info in self.img_info if info[1] == n]
            random.shuffle(lst_tmp)
            lst_tmp = lst_tmp[:img_num]
            new_lst.extend(lst_tmp)
        random.shuffle(new_lst)
        self.img_info = new_lst

    def _get_img_num_per_cls(self):
        """
          依长尾分布计算每个类别应有多少张样本
          :return:
          """
        img_max = len(self.img_info) / self.cls_num
        img_num_per_cls = []
        for cls_idx in range(self.cls_num):
            num = img_max * (self.imb_factor ** (cls_idx / (self.cls_num - 1.0)))  # 列出公式或知道了
            img_num_per_cls.append(int(num))
        return img_num_per_cls

if __name__ == '__main__':
    # todo: 自测代码
    ...