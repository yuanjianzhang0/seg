# 冠状动脉血管造影分割项目(Gradio前端)

#### 北京邮电大学 计算机9组 营员：薛文琪

## 项目概述

本项目旨在使用深度学习模型对冠状动脉血管造影图像进行分割。我们采用了多种主流分割模型架构，包括U-Net, U-Net++, DeepLabV3等。利用PyTorch框架和segmentation_models_pytorch库，可以选择不同的分割模型和编码器进行图像分割任务。

## 目录结构
```
seg/
│
├── model_checkpoints/
│   └── runX/                  # 每次训练的模型权重和记录文件（如 "run1", "run2", ...）
│
├── XCADLabeled/
│   ├── train/
│   │   ├── images/            # 训练图像
│   │   └── masks/             # 训练图像对应的掩码
│   ├── test/
│   │   ├── images/            # 测试图像
│   │   └── masks/             # 测试图像对应的掩码
│   └── predictions/           # 预测结果和指标文件(运行predict_v1.py，计算指标)
│
├── predict.py                 # 预测脚本
├── requirements.txt           # Python依赖项
└── train.py                   # 训练脚本
```

## 环境配置
在运行代码之前，请确保已安装以下依赖项。你可以使用如下命令安装所需的Python包：
```bash
pip install -r requirements.txt
```

`requirements.txt`文件内容：

```
torch
torchvision
pandas
numpy
Pillow
matplotlib
segmentation_models_pytorch
tqdm
gradio
```

## 训练模型

```bash
python train.py
```

### 训练脚本说明
`train.py`脚本包括以下主要步骤：
1. **数据集加载**：从`XCADLabeled/train/images`和`XCADLabeled/train/masks`加载训练图像和掩码。
2. **数据预处理**：调整图像大小并转换为张量。
3. **模型定义**：使用segmentation_models_pytorch库定义分割模型。用户可以通过菜单选择不同的模型架构（如U-Net, U-Net++, DeepLabV3）和编码器。
4. **模型训练**：使用二元交叉熵损失函数和Adam优化器进行模型训练，并保存训练过程中最好的模型权重到`model_checkpoints`文件夹中。
5. **记录训练指标**：保存每个epoch的损失值和评估指标（IoU、F1、Precision、Recall）。

## 进行预测
预测脚本`predict.py`用于对测试图像进行预测。运行以下命令开始预测,之后在浏览器输入http://127.0.0.1:7860
```bash
python predict.py
--->Running on local URL:  http://127.0.0.1:7860
```

### 预测脚本说明
`predict.py`脚本包括以下主要步骤：
1. **数据集加载**：从`XCADLabeled/test/images`和`XCADLabeled/test/masks`加载测试图像和掩码。
2. **数据预处理**：调整图像大小并转换为张量。
3. **加载模型**：加载训练好的分割模型权重。
4. **进行预测**：对测试集中的每张图像进行预测，并保存预测结果到`XCADLabeled/test/predictions`文件夹中。
5. **计算和保存指标**：计算每张图像的评估指标（IoU、F1、Precision、Recall），并将所有图像的指标保存到CSV文件中，同时计算总体的平均值。

## 结果可视化
在预测过程中，预测脚本将保存每张图像的原始图像、真实掩码和预测掩码的可视化结果到`XCADLabeled/test/predictions`文件夹中。

## 使用说明
### 选择模型和编码器
在`train.py`和`predict.py`脚本中，用户可以通过修改以下部分来选择不同的模型架构和编码器：

```python
# 在train.py和predict.py脚本中定义模型部分
model = smp.Unet(
    encoder_name='resnet34',      # 选择编码器，如resnet34, efficientnet-b0等
    encoder_weights='imagenet',   # 使用预训练权重
    in_channels=3,                # 输入通道数
    classes=1                     # 输出通道数
)

# 可替换为其他模型架构，如U-Net++或DeepLabV3等，自行查看
model = smp.UnetPlusPlus(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)

model = smp.DeepLabV3(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)
```
### 编码器（包含ViT）

以下是本项目集成的所有编码器，供君选择

``
'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4'
``


### 示例结果

#### 自建服务器图床，可能有点卡

`https://qianyongdeyu.top/wp-content/uploads/2024/07/metrics_plot.png`

`https://qianyongdeyu.top/wp-content/uploads/2024/07/visualization_12241_33.png.png`

在进行冠状动脉血管造影分割任务测试时，我们使用了U-Net模型架构和ResNet34编码器。训练指标如下：

![](https://qianyongdeyu.top/wp-content/uploads/2024/07/metrics_plot.png)

### 示例配图
以下是对一张测试图像的预测结果可视化：

![](https://qianyongdeyu.top/wp-content/uploads/2024/07/visualization_12241_33.png.png)

## 注意事项

- 请确保数据集文件夹结构与上述目录结构匹配。
- 在进行训练和预测之前，请根据需要调整超参数和文件路径。
- 确保选择合适的模型架构和编码器以获得最佳结果。
