import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from vmamba_unet import VMambaUnet
from color_map_converted import COLOR_MAP
from torchvision import transforms

# 图像预处理类
class Transform:
    def __init__(self, resize_shape=(384, 384)):
        self.resize_shape = resize_shape
        self.geo_transforms = transforms.Compose([
            transforms.Resize((self.resize_shape), interpolation=Image.NEAREST),
            # transforms.RandomHorizontalFlip()
        ])
        self.image_only_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, img):
        img = TF.to_pil_image(img) if not isinstance(img, Image.Image) else img
        img = self.geo_transforms(img)
        img = self.image_only_transforms(img)
        return img

# 加载模型函数
def load_model(model_path, device):
    model = VMambaUnet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 处理单张图像的函数
def process_image(image_path, transform, device):
    """加载并预处理图像"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # 记录原始尺寸
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # 添加批次维度
    return image, original_size

# 推理函数
def predict_image(model, image, device):
    with torch.no_grad():
        output = model(image).argmax(1).squeeze(0).cpu().numpy()  # 获取预测掩码
    return output

# 可视化和保存结果的函数
def visualize_and_save(image_path, pred_mask, output_dir, original_size, color_map):
    """将图像和预测掩码保存到输出文件夹，并保持原始尺寸"""
    # 加载原始图像
    image = Image.open(image_path).convert("RGB")
    
    # 将预测掩码调整为原始尺寸
    pred_color_mask = colorize_mask(pred_mask, color_map)
    pred_color_mask = Image.fromarray(pred_color_mask).resize(original_size, Image.NEAREST)

    # 获取原始文件名
    original_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 保存原始图像
    image_save_path = os.path.join(output_dir, f'{original_filename}_image.png')
    image.save(image_save_path)

    # 保存预测掩码
    pred_mask_save_path = os.path.join(output_dir, f'{original_filename}_pred_mask.png')
    pred_color_mask.save(pred_mask_save_path)

# 颜色映射函数
def colorize_mask(mask, color_map):
    """根据 COLOR_MAP 将预测的 mask 转换为颜色掩码"""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for color, class_index in color_map.items():
        color_mask[mask == class_index] = color
    return color_mask

# 批量推理函数
def run_inference_on_folder(model_path, data_folder, output_dir, transform, device, color_map):
    """对指定文件夹下的所有图像进行推理和可视化，保存原始图像和预测掩码"""
    model = load_model(model_path, device)
    
    # 如果输出文件夹不存在，自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_images = os.listdir(data_folder)
    
    for image_file in all_images:
        image_path = os.path.join(data_folder, image_file)

        # 预处理图像，并记录原始尺寸
        image, original_size = process_image(image_path, transform, device)
        
        # 进行预测
        pred_mask = predict_image(model, image, device)
        
        # 可视化并保存原图和预测掩码
        visualize_and_save(image_path, pred_mask, output_dir, original_size, color_map)
        
    print(f"所有图像的分割结果已保存到 {output_dir}")

# 主程序入口
if __name__ == "__main__":
    # 模型路径
    model_path = "model_pth/model_140.pth"
    
    # 测试图像数据文件夹
    data_folder = "data/entire_img/images"
    
    # 输出结果保存文件夹
    output_dir = "local_visualizations/"
    
    # 使用的设备 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义图像的预处理操作
    transform = Transform(resize_shape=(384, 384))  # 这里使用你训练时的同样预处理操作
    
    # 运行推理过程
    run_inference_on_folder(model_path, data_folder, output_dir, transform, device, COLOR_MAP)
