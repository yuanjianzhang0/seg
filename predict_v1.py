import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置Matplotlib后端
import matplotlib
matplotlib.use('Agg')

checkpoint = torch.load('model_checkpoints/run6/best_model.pth')

model = smp.MAnet(
    encoder_name='vgg19',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)

# 定义数据集类
class BrainTumorTestDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, self.images[idx]

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 数据集和数据加载器
test_dataset = BrainTumorTestDataset('XCADLabeled/test/images', 'XCADLabeled/test/masks', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载训练好的模型权重

model.load_state_dict(checkpoint)
model.eval()

# 获取唯一的运行索引
run_index = 1
while os.path.exists(f'detect/run{run_index}'):
    run_index += 1
output_dir = f'detect/run{run_index}'
os.makedirs(output_dir, exist_ok=True)

# 创建保存预测结果的目录
predictions_dir = os.path.join(output_dir, 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

# 定义评价指标
def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(y_true * y_pred)
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - tp
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return precision, recall, iou, f1

# 仿真图
def visualize(image, mask, prediction, save_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[1].imshow(mask.squeeze(), cmap="gray")
    ax[1].set_title("Mask")
    ax[2].imshow(prediction.squeeze(), cmap="gray")
    ax[2].set_title("Prediction")
    plt.savefig(save_path)
    plt.close(fig)

# 推理并保存结果
metrics = []
with torch.no_grad():
    for images, masks, image_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu().numpy()
        outputs = (outputs > 0.5).astype(np.uint8)

        masks = masks.cpu().numpy()
        masks = (masks > 0.5).astype(np.uint8)

        for output, mask, image_name in zip(outputs, masks, image_names):
            output_image = Image.fromarray(output[0] * 255)
            output_image = output_image.resize((images.shape[3], images.shape[2]))  # 调整到原始大小
            output_image.save(f'{predictions_dir}/{image_name}')

            # 计算指标
            precision, recall, iou, f1 = calculate_metrics(mask, output)
            metrics.append((image_name, precision, recall, iou, f1))

            # 保存可视化结果
            visualize(images[0].cpu(), masks[0], output[0], f'{predictions_dir}/visualization_{image_name}.png')

# 保存所有图片的预测指标
metrics_df = pd.DataFrame(metrics, columns=['Image', 'Precision', 'Recall', 'IoU', 'F1'])
metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)

# 计算总体指标
mean_metrics = metrics_df[['Precision', 'Recall', 'IoU', 'F1']].mean(axis=0)
with open(f'{output_dir}/overall_metrics.txt', 'w') as f:
    f.write(f'Mean Precision: {mean_metrics["Precision"]}\n')
    f.write(f'Mean Recall: {mean_metrics["Recall"]}\n')
    f.write(f'Mean IoU: {mean_metrics["IoU"]}\n')
    f.write(f'Mean F1: {mean_metrics["F1"]}\n')

print(f"推理完成，结果已保存到 '{output_dir}' 目录中。")
