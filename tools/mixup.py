import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def mixup_data(x, y, alpha=1.0, device=None):
    if device is None:
        device = x.device

    # 通过beta分布获得lambda，beta分布的参数alpha==beta，只设置alpha
    lam = np.random.beta(alpha, alpha) if alpha > 1 else 1

    # 获取需要混叠的图片数量
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # mixup
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    path_1 = "../../images/cat_4093.jpg"
    path_2 = "../../images/dog_10770.jpg"

    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)
    img_1 = cv2.resize(img_1, (224, 224))
    img_2 = cv2.resize(img_2, (224, 224))

    alpha = 1
    figsize = 15
    plt.figure(figsize=(figsize, figsize))
    for i in range(1, 10):
        lam = np.random.beta(alpha, alpha)
        im_mixup = (img_1 * lam + img_2 * (1 - lam)).astype(np.uint8)
        im_mixup = cv2.cvtColor(im_mixup, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 3, i)
        plt.title(f"lambda_{lam:.2f}")
        plt.imshow(im_mixup)

    plt.show()