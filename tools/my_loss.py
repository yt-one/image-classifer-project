import torch
from torch import nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, prediction, target):
        # 1. softmax
        # 2. 制作权重，真实类别的权重为 1-smoothing, 其余类别权重为 (smoothing) / (K-1)
        # 3. 依据交叉熵损失公式计算loss

        log_prob = F.log_softmax(prediction, dim=-1)

        weight = prediction.new_ones(prediction.size()) * self.smoothing / (prediction.size(-1) - 1.)

        # scatter_(dim, index, src) 按照索引 index 在维度 dim 上，用 src 的值填充目标张量。
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))

        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

if __name__ == "__main__":
    output = torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    criterion = LabelSmoothingLoss(0.001)
    loss = criterion(output, label)

    print(f"CrossEntropy: {loss}")