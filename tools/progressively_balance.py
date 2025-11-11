from collections import Counter

import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

from datasets.cifar_longtail import CifarLTDataset
from tools.common_tools import check_data_dir


# epoch -> sampler -> Dataloder
# func(epoch) -> sampler

class ProgressiveSampler:
    def __init__(self, dataset, max_epoch, q=0.5):
        """
        :param dataset: PyTorch Dataset，需包含属性 train_targets（类别标签）
        :param max_epoch: 总训练轮数 T
        :param q: 控制采样平衡程度的超参数 ∈ [0, 1]
        """
        self.dataset = dataset
        self.train_targets = np.array([label for _, label in dataset.img_info])
        self.max_epoch = max_epoch
        self.q = q
        self.num_classes = len(np.unique(self.train_targets))
        self.nj = np.bincount(self.train_targets)

        # 预先计算两种分布
        self.p_ib = self._cal_class_prob(q=1.0)  # 不平衡分布
        self.p_cb = self._cal_class_prob(q=0.0)  # 完全平衡分布

    def _cal_class_prob(self, q):
        """
        根据公式 (1)：p_j = n_j^q / sum_i n_i^q
        """
        prob = self.nj ** q
        prob = prob / prob.sum()
        return prob

    def _cal_pb_prob(self, t):
        """
        根据公式 (2)：p_pb(t) = (1 - t/T) * p_ib + (t/T) * p_cb
        """
        ratio = t / self.max_epoch
        p_pb = (1 - ratio) * self.p_ib + ratio * self.p_cb

        # p_pb /= self.nj  # 非常重要！ pytorch 是样本权重
        return p_pb

    def __call__(self, epoch):
        """
        根据当前 epoch 生成 WeightedRandomSampler
        """
        P_pb = self._cal_pb_prob(t=epoch)
        P_pb = torch.tensor(P_pb, dtype=torch.float)

        # 为每个样本分配对应类别的采样权重， p_pb: 每个类的采样权重
        samples_weights = P_pb[self.train_targets]

        # 生成采样器
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True
        )
        return sampler, P_pb

    def plot_line(self):
        """
        绘制采样概率变化曲线
        """
        epochs = list(range(self.max_epoch + 1))
        probs = [self._cal_pb_prob(t) for t in epochs]
        probs = np.stack(probs)

        plt.figure(figsize=(8, 5))
        for i in range(self.num_classes):
            plt.plot(epochs, probs[:, i], label=f'class {i}')
        plt.xlabel("Epoch")
        plt.ylabel("Sampling Probability")
        plt.title("Progressively-Balanced Sampling Evolution")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 路径准备
    train_dir = r"D:\Learn\Datasets\cifar-10-python\cifar10_train"
    check_data_dir(train_dir)

    a_transform = transforms.Compose([transforms.ToTensor()])
    train_data = CifarLTDataset(root_dir=train_dir, transform=a_transform, isTrain=True)

    max_epoch = 200
    sampler_generator = ProgressiveSampler(train_data, max_epoch)
    sampler_generator.plot_line()

    for epoch in range(max_epoch):
        if epoch % 20 != 19:
            continue

        sampler, _ = sampler_generator(epoch)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            sampler=sampler
        )

        labels = []
        for data in train_loader:
            _, label = data
            labels.extend(label.tolist())

        print(f"Epoch: {epoch}, Counter: {Counter(labels)}")
