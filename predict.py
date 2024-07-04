import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# 加载模型
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=3,
    classes=1
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载训练好的模型权重
checkpoint = torch.load('model_checkpoints/run3/best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# 定义分割函数
def segment_brain_tumor(image):
    image = Image.fromarray(image)
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        output = torch.sigmoid(output)
        output = output.cpu().numpy()[0, 0]
        output = (output > 0.5).astype(np.uint8) * 255

    return output


# 创建 Gradio 界面
iface = gr.Interface(
    fn=segment_brain_tumor,
    inputs=gr.Image(type="numpy", label="上传图像"),
    outputs=gr.Image(type="numpy", label="分割结果"),
    title="管状动脉分割 / 薛文琪",
    description="上传一张图像以获得分割结果。"
)

# 启动界面
iface.launch()
