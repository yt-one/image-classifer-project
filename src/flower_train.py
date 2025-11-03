import argparse
import os
import pickle
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 得到当前py文件所在的目录，不管在哪里运行
sys.path.append(os.path.join(BASE_DIR, '..'))  # 添加自定义包的路径

from datetime import datetime
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from datasets.flower_102 import FlowerDataset
from tools.common_tools import setup_seed, check_data_dir, Logger, show_confMat, plot_line
from tools.model_trainer import ModelTrainer
from config.flower_config import cfg

setup_seed(12345) # 先固定随机种子
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--lr", default=None, type=float, help="learning rate")
parser.add_argument('--bs', default=None, type=int, help="training batch size")
parser.add_argument("--max_epoch", default=None, type=int)
parser.add_argument("--data_root_dir", default=r"D:\Learn\Datasets\flowers102", type=str, help="path to your dataset")

args = parser.parse_args()

cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

if __name__ == '__main__':
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%m-%d_%H-%M")
    log_dir = os.path.join(BASE_DIR, "..", "..", "results", time_str) # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_dir = os.path.join(args.data_root_dir, "train")
    valid_dir = os.path.join(args.data_root_dir, "valid")
    check_data_dir(train_dir)  # 检查数据路径是否存在，不存在提前报错
    check_data_dir(valid_dir)

    # 创建log文件夹
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%m-%d_%H-%M")
    log_dir = os.path.join(BASE_DIR, "..", "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()

    # 1. 数据
    train_data = FlowerDataset(root_dir=train_dir, transform=cfg.transforms_train)
    valid_data = FlowerDataset(root_dir=valid_dir, transform=cfg.transforms_valid)


    # 2. 模型
    model = resnet18()
    path_state_dict = r"D:\Learn\Datasets\flowers102\pretrained_model\resnet18-f37072fd.pth"
    if os.path.exists(path_state_dict):
        pretrained_state_dict = torch.load(path_state_dict, map_location="cuda:0")
        model.load_state_dict(pretrained_state_dict)
        logger.info(f"load pretrained model!")
    else:
        logger.info(f"the pretrained model path {path_state_dict} does not exist!")

    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.cls_num)  # 102
    model.to(device)

    # 3. 损失函数，优化器
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.factor)


    # 4. 迭代训练
    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(valid_data, batch_size=cfg.valid_bs, shuffle=False, num_workers=cfg.workers)

    # 记录训练所使用的模型、损失函数、优化器、调度器、配置参数 cfg
    logger.info(
        "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
            cfg, loss_f, scheduler, optimizer, model
        )
    )  # todo: cfg需要实现字符串方法

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(cfg.max_epoch):
        # dataloader

        # train
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(train_loader, model, loss_f, optimizer,
                                                                                epoch, device, cfg.log_interval, cfg.max_epoch,
                                                                                logger)

        # valid
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(valid_loader, model, loss_f, device, logger)

        logger.info(
            "Epoch[{:<3d}/{:<3d}] Train Acc:{:.2%} Valid Acc:{:.2%} "
            "Train loss:{:.4f} Valid loss:{:.4f} LR:{:.6f}".format(
                epoch + 1,
                cfg.max_epoch,
                acc_train,
                acc_valid,
                loss_train,
                loss_valid,
                optimizer.param_groups[0]["lr"]
            )
        )

        # 学习率更新
        scheduler.step()

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epochs=epoch,
                     verbose=epoch == cfg.max_epoch)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epochs=epoch,
                     verbose=epoch == cfg.max_epoch)

        # 保存loss曲线，acc曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        # 模型保存
        # 保存模型
        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
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
                if epoch == cfg.max_epoch - 1
                else "checkpoint_best.pkl"
            )

            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

            # 保存错误图片的路径
            err_ims_name = f"error_imgs_{epoch}.pkl" if epoch == cfg.max_epoch - 1 else f"error_imgs_best.pkl"
            path_err_imgs = os.path.join(log_dir, err_ims_name)
            error_info = {"train": path_error_train, "valid": path_error_valid}
            pickle.dump(error_info, open(path_err_imgs, "wb"))

        logger.info("{} done, best acc: {} in : {}".format(
            datetime.strftime(datetime.now(), "%m-%d_%H-%M"), best_acc, best_epoch
        ))