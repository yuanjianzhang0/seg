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


### 示例结果

#### 图在国内小水管阿里云服务器，可能有点卡，请老师见谅

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
