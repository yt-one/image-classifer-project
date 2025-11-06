import logging
import random
import numpy as np
import torch
import os

from matplotlib import pyplot as plt
from torch import nn
from torchvision.models import resnet18, vgg16_bn
from models.se_resnet import se_resnet50


def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # 保证结果可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_data_dir(data_dir):
    assert os.path.exists(data_dir), \
        f"\n\n路径不存在，当前变量中指定的路径是：\n{os.path.abspath(data_dir)}\n请检查相对路径的设置，或者文件是否存在"

def show_confMat(confusion_mat, classes, set_name, out_dir, epochs, verbose=False, figsize=None, perc=False):
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num - 10]
    plt.figure(figsize=(int(figsize), int(figsize * 1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=69)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Product label')
    plt.ylabel('True label')
    plt.title('Confusion_matrix_{}_{}'.format(set_name, epochs))

    # 打印数字
    if perc:
        cls_per_num = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_num
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]),
                         va='center', ha='center', fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]),
                         va='center', ha='center', color='red')

    # 保存
    plt.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2f} Precision: {:.2f}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))
            ))

def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    plt.plot(train_x, train_y, label="Train")
    plt.plot(valid_x, valid_y, label="Valid")

    plt.ylabel(str(mode))
    plt.xlabel("Epoch")

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


class Logger:
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置控制台Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger



def get_model(cfg, cls_num, logger):
    model_name = cfg.model_name.lower()  # 从配置中取模型名
    logger.info(f"Creating model: {model_name}")

    if model_name == "resnet18":
        model = resnet18(num_classes=cls_num)
        path_state_dict = cfg.path_resnet18

        if os.path.exists(path_state_dict):
            pretrained_state_dict = torch.load(path_state_dict)
            model.load_state_dict(pretrained_state_dict)
            logger.info(f"load pretrained model!")
        else:
            logger.info(f"the pretrained model path {path_state_dict} does not exist!")

        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cls_num)

    elif model_name == "vgg16_bn":
        model = vgg16_bn(num_classes=cls_num)
        path_state_dict = cfg.path_vgg16_bn

        if os.path.exists(path_state_dict):
            pretrained_state_dict = torch.load(path_state_dict)
            model.load_state_dict(pretrained_state_dict)
            logger.info(f"load pretrained model!")
        else:
            logger.info(f"the pretrained model path {path_state_dict} does not exist!")

        # 修改最后一层
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, cls_num)

    elif model_name == 'se_resnet50':
        model = se_resnet50(num_classes=cls_num)
        path_state_dict = cfg.path_se_res50

        if os.path.exists(path_state_dict):
            pretrained_state_dict = torch.load(path_state_dict)
            model.load_state_dict(pretrained_state_dict)
            logger.info(f"load pretrained model!")
        else:
            logger.info(f"the pretrained model path {path_state_dict} does not exist!")

        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, cls_num)
    else:
        raise ValueError(f"Unsupported model: {model_name}")



    return model