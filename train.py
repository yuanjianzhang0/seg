import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from modelset import model
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
# 定义模型

num_epochs = 60
# 定义数据集类
class BrainTumorDataset(Dataset):
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

        return image, mask

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 数据集和数据加载器
train_dataset = BrainTumorDataset('XCADLabeled/train/images', 'XCADLabeled/train/masks', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 损失函数和优化器
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 自动获取模型细节
model_details = {
    "Author" : "Wenqi Xue",
    "Team" : "AIoTMaster"
}

# 查找新的运行文件夹
run_id = 1
while os.path.exists(f"model_checkpoints/run{run_id}"):
    run_id += 1
run_folder = f"model_checkpoints/run{run_id}"
os.makedirs(run_folder)

with open(os.path.join(run_folder, "model_details.json"), "w") as f:
    json.dump(model_details, f, indent=4)

# 训练模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_iou = 0.0

metrics_history = {'loss': [], 'iou': [], 'f1': [], 'precision': [], 'recall': []}

def calculate_metrics(preds, targets):
    preds = preds > 0.5
    targets = targets > 0.5

    TP = (preds & targets).sum().item()
    FP = (preds & ~targets).sum().item()
    FN = (~preds & targets).sum().item()
    TN = (~preds & ~targets).sum().item()

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = TP / (TP + FP + FN + 1e-7)

    return precision, recall, f1, iou

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    metrics = {'iou': [], 'f1': [], 'precision': [], 'recall': []}

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu()
            targets = masks.cpu()
            precision, recall, f1, iou = calculate_metrics(preds, targets)

            metrics['iou'].append(iou)
            metrics['f1'].append(f1)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)

            pbar.set_postfix({
                'loss': epoch_loss / len(train_loader),
                'iou': np.mean(metrics['iou']),
                'f1': np.mean(metrics['f1']),
                'precision': np.mean(metrics['precision']),
                'recall': np.mean(metrics['recall']),
            })
            pbar.update(1)

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_iou = np.mean(metrics['iou'])
    avg_f1 = np.mean(metrics['f1'])
    avg_precision = np.mean(metrics['precision'])
    avg_recall = np.mean(metrics['recall'])

    metrics_history['loss'].append(avg_epoch_loss)
    metrics_history['iou'].append(avg_iou)
    metrics_history['f1'].append(avg_f1)
    metrics_history['precision'].append(avg_precision)
    metrics_history['recall'].append(avg_recall)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}, IoU: {avg_iou}, F1: {avg_f1}, Precision: {avg_precision}, Recall: {avg_recall}")

    # 保存表现最好的模型
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), os.path.join(run_folder, "best_model.pth"))

# 保存最后一个epoch的模型
torch.save(model.state_dict(), os.path.join(run_folder, "last_model.pth"))

print("训练完成！")

# 保存指标到文件
metrics_file = os.path.join(run_folder, 'metrics_history.json')
with open(metrics_file, 'w') as f:
    json.dump(metrics_history, f, indent=4)

# 绘制指标图表
def plot_metrics(metrics_history, save_path):
    epochs = range(1, len(metrics_history['loss']) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_history['precision'], label='Precision')
    plt.plot(epochs, metrics_history['recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision and Recall')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_history['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics_history['iou'], label='IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

plot_metrics(metrics_history, os.path.join(run_folder, 'metrics_plot.png'))
