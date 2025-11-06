from torchvision import transforms

class FlowerConfig:
    pass

cfg = FlowerConfig()

# cfg.model_name = 'resnet18'
cfg.model_name = 'vgg16_bn'
# cfg.model_name = 'se_resnet50'

# 预训练参数

# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-600232d0.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
# SE-ResNet50: https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl

cfg.path_resnet18 = ...
cfg.path_vgg16_bn = ...
cfg.path_se_res50 = ...

cfg.train_bs = 32
cfg.valid_bs = 64
cfg.workers = 0

cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1
cfg.milestones = [30, 45]
cfg.max_epoch = 50

cfg.log_interval = 10

norm_mean = [0.485, 0.456, 0.406] # imagenet 120万图像统计得来
norm_std = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(norm_mean, norm_std)

cfg.transforms_train =  transforms.Compose([
        transforms.Resize((256,)),  # 最短边256
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform,
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform,
])