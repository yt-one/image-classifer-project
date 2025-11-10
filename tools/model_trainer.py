from collections import Counter
import numpy as np
import torch

from config.flower_config import cfg
from tools.mixup import mixup_data, mixup_criterion


class ModelTrainer:
    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_idx, device, log_interval, max_epoch, logger):
        model.train()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        label_list = []

        for i, data in enumerate(data_loader):
            # _, labels = data
            inputs, labels, path_imgs = data
            label_list.extend(labels.tolist())

            # inputs, labels, path_imgs = data
            # inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            if cfg.mixup:
                mixed_inputs, label_a, label_b, lam = mixup_data(inputs, labels, cfg.mixup_alpha, device)
                inputs = mixed_inputs

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()
            if cfg.mixup:
                loss = mixup_criterion(loss_f, outputs, label_a, label_b, lam) #  # outputs.cpu(), labels.cpu()
            else:
                loss = loss_f(outputs, labels)
                
            loss.backward()
            optimizer.step()

            # 计算 loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)

            # 预测结果
            _, predicted = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

                # 记录错误样本的信息
                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i, path_imgs[j]))

            acc_avg = conf_mat.trace() / conf_mat.sum()

            # 输出 iteration 级训练信息
            if i % log_interval == log_interval - 1:
                logger.info(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}".format(
                        epoch_idx + 1, max_epoch, i + 1, len(data_loader), loss_mean, acc_avg
                    )
                )
        logger.info("epoch{} sampler: {}".format(epoch_idx, Counter(label_list)))
        return loss_mean, acc_avg, conf_mat, path_error

    @staticmethod
    def valid(data_loader, model, loss_f, device, logger):
        model.eval()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        path_error = []

        for i, data in enumerate(data_loader):
            inputs, label, _ = data
            inputs, label= inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs, label)
            # 计算 loss
            loss_sigma.append(loss.item())

            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(label)):
                cate_i = label[j].cpu().numpy()
                pre_i = (predicted[j].cpu().numpy())
                conf_mat[cate_i, pre_i] += 1.

        # 计算平均损失与准确率
        loss_mean = np.mean(loss_sigma)
        acc_avg = conf_mat.trace() / conf_mat.sum()

        logger.info("Valid: Loss:{:.4f} Acc:{:.2%}".format(loss_mean, acc_avg))
        return loss_mean, acc_avg, conf_mat, path_error