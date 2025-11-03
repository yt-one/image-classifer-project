import sys
import os
from datetime import datetime

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from image_cls.datasets.flower_102 import FlowerDataset
from image_cls.tools.model_trainer import ModelTrainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 得到当前py文件所在的目录，不管在哪里运行
sys.path.append(os.path.join(BASE_DIR, '..'))
print(BASE_DIR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%m-%d_%H-%M")
    log_dir = os.path.join(BASE_DIR, "..", "..", "results", time_str) # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_dir = r"D:\Learn\Datasets\flowers102\train"
    valid_dir = r"D:\Learn\Datasets\flowers102\valid"
    workers = 0

    train_bs = 64
    valid_bs = 64

    lr_init = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    factor = 0.1
    milestones = [30, 45]  # 什么时候降学习率
    max_epoch = 50

    log_interval = 10  # 日志打印了间隔

    # 1. 数据
    norm_mean = [0.485, 0.456, 0.406]  # imagenet 120万张图片统计得来
    norm_std = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(norm_mean, norm_std)

    transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform]
    )

    transforms_valid = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         normTransform]
    )

    train_data = FlowerDataset(root_dir=train_dir, transform=transforms_train)
    valid_data = FlowerDataset(root_dir=valid_dir, transform=transforms_valid)

    # 2. 模型
    model = resnet18()
    path_state_dict = r"D:\Learn\Datasets\flowers102\pretrained_model\resnet18-f37072fd.pth"
    if os.path.exists(path_state_dict):
        pretrained_state_dict = torch.load(path_state_dict, map_location="cuda:0")
        model.load_state_dict(pretrained_state_dict)

    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.cls_num)  # 102
    model.to(device)

    # 3. 损失函数，优化器
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=factor)

    # 4. 迭代训练
    train_loader = DataLoader(train_data, batch_size=train_bs, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(valid_data, batch_size=valid_bs, shuffle=False, num_workers=workers)

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(max_epoch):
        # dataloader

        # train
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(train_loader, model, loss_f, optimizer,
                                                                                epoch, device, log_interval, max_epoch)

        # valid
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(valid_loader, model, loss_f, device)

        print(
            "Epoch[{:<3d}/{:<3d}] Train Acc:{:.2%} Valid Acc:{:.2%} "
            "Train loss:{:.4f} Valid loss:{:.4f} LR:{:.6f}".format(
                epoch + 1,
                max_epoch,
                acc_train,
                acc_valid,
                loss_train,
                loss_valid,
                optimizer.param_groups[0]["lr"]
            )
        )

        # 学习率更新
        scheduler.step()

        # 模型保存
        # 保存模型
        if best_acc < acc_valid or epoch == max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc
            }

            pkl_name = (
                "checkpoint_{}.pkl".format(epoch)
                if epoch == max_epoch - 1
                else "checkpoint_best.pkl"
            )

            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)