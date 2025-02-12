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

# 计算每个类别的 IoU 和 Precision 函数
def compute_metrics(pred_mask, true_mask, num_classes):
    """计算每个类别的 IoU 和 Precision"""
    ious = []
    precisions = []
    
    # 检查预测掩码和真实掩码的唯一值和形状
    print(f"Predicted mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")
    print(f"True mask shape: {true_mask.shape}, unique values: {np.unique(true_mask)}")
    
    for cls in range(num_classes):
        pred_class = (pred_mask == cls)
        true_class = (true_mask == cls)
        
        # 计算交集和并集
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()

        # 防止分母为零
        iou = intersection / union if union != 0 else 0.0
        ious.append(iou)

        # 精度计算
        tp = intersection
        fp = pred_class.sum() - tp
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        precisions.append(precision)

        # 输出每个类别的调试信息
        print(f"Class {cls}:")
        print(f"Predicted mask for class {cls}: {pred_class.sum()} pixels")
        print(f"True mask for class {cls}: {true_class.sum()} pixels")
        print(f"Intersection (TP) for class {cls}: {intersection} pixels")
        print(f"Union for class {cls}: {union} pixels")
        print(f"IoU for class {cls}: {iou:.4f}")
        print(f"Precision for class {cls}: {precision:.4f}")

    return ious, precisions



    return ious

# 加载和处理真实掩码的函数
def process_true_mask(mask_path, resize_shape):
    """加载并调整真实掩码的尺寸"""
    true_mask = Image.open(mask_path)
    true_mask = true_mask.resize(resize_shape, Image.NEAREST)  # 调整为与输入图像相同的尺寸
    true_mask = np.array(true_mask)  # 转换为 NumPy 数组
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"True mask shape: {true_mask.shape}, unique values: {np.unique(true_mask)}")
    print("--------------------------------------------------------------")
    return true_mask

# 批量推理函数，加入 IoU 和 Precision 计算
def run_inference_on_folder_with_metrics(model_path, data_folder, true_mask_folder, output_dir, transform, device, color_map, num_classes):
    """对指定文件夹下的所有图像进行推理、计算 IoU 和 Precision，并保存可视化结果"""
    model = load_model(model_path, device)
    
    # 如果输出文件夹不存在，自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_images = os.listdir(data_folder)
    total_iou_per_class = np.zeros(num_classes)  # 初始化每个类别的 IoU 累计
    total_precision_per_class = np.zeros(num_classes)  # 初始化每个类别的 Precision 累计
    num_images = len(all_images)
    
    for image_file in all_images:
        image_path = os.path.join(data_folder, image_file)
        true_mask_path = os.path.join(true_mask_folder, image_file.replace(".jpg", "_mask.png"))  # 真实掩码路径

        # 预处理图像，并记录原始尺寸
        image, original_size = process_image(image_path, transform, device)
        
        # 加载并处理真实掩码
        true_mask = process_true_mask(true_mask_path, resize_shape=(384, 384))  # 调整真实掩码为与预测掩码相同的尺寸
        true_mask -= 2  # 将索引值全部减去2

        # 进行预测
        pred_mask = predict_image(model, image, device)
        
        # 可视化并保存原图和预测掩码
        visualize_and_save(image_path, pred_mask, output_dir, original_size, color_map)
        
        # 计算该图像的 IoU 和 Precision
        iou_per_class, precision_per_class = compute_metrics(pred_mask, true_mask, num_classes)
        total_iou_per_class += np.nan_to_num(iou_per_class)  # 累加 IoU，忽略 NaN
        total_precision_per_class += np.nan_to_num(precision_per_class)  # 累加 Precision，忽略 NaN
    
    # 计算平均 IoU 和 Precision
    avg_iou_per_class = total_iou_per_class / num_images
    avg_precision_per_class = total_precision_per_class / num_images

    # 输出每个类别的 IoU 和 Precision
    for class_index, (iou, precision) in enumerate(zip(avg_iou_per_class, avg_precision_per_class)):
        print(f"Class {class_index} - IoU: {iou:.4f}, Precision: {precision:.4f}")
    
    print(f"所有图像的分割结果和 IoU、Precision 已保存到 {output_dir}")

    
    # 计算平均 IoU
    avg_iou_per_class = total_iou_per_class / num_images
    for class_index, iou in enumerate(avg_iou_per_class):
        print(f"Class {class_index} - IoU: {iou:.4f}")
    
    print(f"所有图像的分割结果和 IoU 已保存到 {output_dir}")


# 主程序入口
if __name__ == "__main__":
    # 模型路径
    model_path = "model_pth/model_90.pth"
    
    # 测试图像数据文件夹
    data_folder = "data/entire_img/images"
    
    # 真实掩码文件夹
    true_mask_folder = "data/entire_img/masks"
    
    # 输出结果保存文件夹
    output_dir = "local_visualizations/"
    
    # 使用的设备 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义图像的预处理操作
    transform = Transform(resize_shape=(384, 384))  # 这里使用你训练时的同样预处理操作
    
    # 类别数量
    num_classes = len(COLOR_MAP)
    print(f"Number of classes: {num_classes}")

    # 运行推理过程，并计算 IoU 和 Precision
    run_inference_on_folder_with_metrics(model_path, data_folder, true_mask_folder, output_dir, transform, device, COLOR_MAP, num_classes)

