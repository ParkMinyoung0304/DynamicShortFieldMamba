import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
# 导入imagefolder
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from vmamba_unet import VMambaUnet  # Assuming VMambaUnet is suitable for segmentation
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import random
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
from color_map_converted import COLOR_MAP 
import numpy as np

########################################################---------Start:数据增强部分-----------##########################################################
def unnormalize(image):
    return image * 0.5 + 0.5  # 逆归一化操作，适用于均值为0.5，标准差为0.5的情况

class Transform:
    def __init__(self, resize_shape=(384, 384)):
        self.resize_shape = resize_shape  # 根据您的图像尺寸调整
        self.geo_transforms = transforms.Compose([
            transforms.Resize((self.resize_shape), interpolation=Image.NEAREST),  # 调整为更适合长方形图像的尺寸
            # transforms.RandomCrop((384, 384)),  # 根据需要调整裁剪尺寸
            transforms.RandomHorizontalFlip()  # 只使用水平翻转
        ])
        
        self.image_only_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # 转换掩模为张量
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)

        img = TF.to_pil_image(img) if not isinstance(img, Image.Image) else img
        # 确保掩码是uint8类型再转换为PIL图像
        mask = mask.byte() if isinstance(mask, torch.Tensor) else mask
        mask = TF.to_pil_image(mask) if not isinstance(mask, Image.Image) else mask

        img = self.geo_transforms(img)
        mask = self.geo_transforms(mask)
        img = self.image_only_transforms(img)
        mask = self.to_tensor(mask)  # 直接转换掩模为张量，不应用归一化

        return img, mask
########################################################---------Ending:数据增强-----------##########################################################

########################################################---------Start:重构Dataset类-----------########################################################
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_names, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        mask_path = os.path.join(self.mask_dir, self.file_names[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # 假设 mask 是 RGB 图像

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = self.color_to_class(np.array(mask))

        return image, torch.tensor(mask, dtype=torch.long)

    def color_to_class(self, mask):
        """Convert a color mask (numpy array) to a class index mask."""
        if mask.shape[0] == 3 and len(mask.shape) == 3:  # 假设通道数为3，且是第一个维度
            mask = mask.transpose(1, 2, 0)  # 将形状从 (3, 224, 224) 转换为 (224, 224, 3)
        mask = (mask * 255).astype(np.uint8)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
        for color, class_index in COLOR_MAP.items():
            color_array = np.array(color, dtype=np.uint8).reshape(1, 1, 3)  # 将颜色数组形状调整为 (1, 1, 3)
            # print("color_array:",color_array)
            matches = (mask == color_array).all(axis=-1)  # 检查颜色是否匹配
            
            class_mask[matches] = class_index # 将匹配的像素设置为类别索引
        return class_mask

def load_dataset(image_dir, mask_dir, transform):
    all_images = os.listdir(image_dir)
    train_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)
    train_dataset = SegmentationDataset(image_dir, mask_dir, train_files, transform)
    test_dataset = SegmentationDataset(image_dir, mask_dir, test_files, transform)
    
    # 返回训练和测试的 DataLoader 以及测试集的完整数据集
    return DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2), DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2), test_dataset


########################################################---------Ending:重构Dataset类-----------######################################################

########################################################---------Start:图像分割结果可视化-----------###############################################

def colorize_mask(mask, color_map):
    """根据COLOR_MAP将预测的mask转换为颜色掩码"""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for color, class_index in color_map.items():
        color_mask[mask == class_index] = color  # 将每个类的像素替换为对应的颜色
    return color_mask

def visualize_all_predictions(model, dataset, output_dir, epoch, device='cpu'):
    """可视化并单独保存所有测试集的分割结果，使用原文件名作为前缀，并将颜色掩码覆盖到原图上"""
    print(f"开始保存所有测试图片的预测结果 (Epoch {epoch})")
    model.eval()
    
    # 创建用于保存可视化结果的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_classes = len(COLOR_MAP) + 1  # 类别数
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, true_mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备
            true_mask = true_mask.cpu().numpy()  # 将真实掩码转换为NumPy数组
            pred_mask = model(image).argmax(1).squeeze(0).cpu().numpy()  # 获取预测掩码
            image = unnormalize(image.squeeze(0)).permute(1, 2, 0).cpu().numpy()  # 逆归一化后的图像

            # 获取文件名作为前缀，去掉扩展名
            original_filename = os.path.splitext(dataset.file_names[idx])[0]

            # 将预测掩码和真实掩码转换为彩色掩码
            true_color_mask = colorize_mask(true_mask, COLOR_MAP)
            pred_color_mask = colorize_mask(pred_mask, COLOR_MAP)

            # 创建图像可视化
            fig, ax = plt.subplots(figsize=(5, 5))

            # 保存原始图像
            ax.imshow(image)
            ax.axis('off')
            plt.tight_layout()
            image_save_path = os.path.join(output_dir, f'{original_filename}_image_epoch_{epoch}.png')
            plt.savefig(image_save_path)
            plt.close()

            # 保存真实掩码
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(true_color_mask)
            ax.axis('off')
            plt.tight_layout()
            true_mask_save_path = os.path.join(output_dir, f'{original_filename}_true_mask_epoch_{epoch}.png')
            plt.savefig(true_mask_save_path)
            plt.close()

            # 保存预测掩码
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(pred_color_mask)
            ax.axis('off')
            plt.tight_layout()
            pred_mask_save_path = os.path.join(output_dir, f'{original_filename}_pred_mask_epoch_{epoch}.png')
            plt.savefig(pred_mask_save_path)
            plt.close()

            # 保存差异掩码
            fig, ax = plt.subplots(figsize=(5, 5))
            diff = (true_mask != pred_mask).astype(np.uint8)  # 计算差异掩码
            ax.imshow(diff, cmap='gray')
            ax.axis('off')
            plt.tight_layout()
            diff_save_path = os.path.join(output_dir, f'{original_filename}_diff_mask_epoch_{epoch}.png')
            plt.savefig(diff_save_path)
            plt.close()

    print(f"所有测试图片的预测结果已单独保存完毕 (Epoch {epoch})")


########################################################---------Ending:图像分割结果可视化-----------#############################################

########################################################---------Start:参数计算-----------#################################################################
def calculate_per_class_iou(pred, target, num_classes):
    """计算每个类别的 IoU，并返回所有类别的 IoU 列表以及平均 mIoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # 创建一个字典来保存每个类别的 IoU 值
    class_ious = {}

    for cls in range(num_classes):
        pred_inds = pred == cls  # 预测结果中属于当前类别的像素
        target_inds = target == cls  # 真实结果中属于当前类别的像素
        intersection = (pred_inds & target_inds).sum().item()  # 交集：预测和真实都属于当前类别的像素
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection  # 并集：预测或真实属于当前类别的像素
        if union == 0:
            iou = float('nan')  # 如果并集为 0，IoU 设置为 NaN
        else:
            iou = intersection / union  # IoU = 交集 / 并集

        ious.append(iou)
        class_ious[cls] = iou  # 将每个类别的 IoU 存储到字典中

    # 计算所有有效类别的平均 mIoU
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else float('nan')

    return class_ious, miou  # 返回每个类别的 IoU 以及平均 mIoU


########################################################---------Ending:参数计算-----------#################################################################

transform = Transform()
# Load datasets
train_loader_entire, test_loader_entire, test_dataset_entire = load_dataset('data/entire_img/images', 'data/entire_img/masks', transform)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VMambaUnet().to(device)
    # model = resnet50().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # TensorBoard logger
    writer = SummaryWriter("./logs")
    save_dir = "model_pth"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training loop
    total_train_step = 0
    total_test_step = 0
    num_epochs = 150
    for i in range(num_epochs):
        train_loader, test_loader = train_loader_entire, test_loader_entire

        print(f"--------第{i+1}轮训练开始------------")
        # Training phase
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        
        # 测试步骤
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        total_iou = 0
        class_iou_dict = {cls: 0 for cls in range(len(COLOR_MAP))}  # 初始化类别 IoU 字典

        with torch.no_grad():
            for images, labels in test_loader_entire:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                total_test_loss += loss.item()

                predictions = outputs.argmax(1)
                accuracy = (predictions == labels).float().mean().item()
                total_accuracy += accuracy

                # 计算每个类别的IoU和平均mIoU
                per_class_iou, miou = calculate_per_class_iou(predictions, labels, num_classes=len(COLOR_MAP))
                # total_iou += miou  # 将当前批次的mIoU累加

                # 将每一批次的 IoU 结果累加到总的类别 IoU 结果中
                for cls in per_class_iou:
                    if not np.isnan(per_class_iou[cls]):
                        class_iou_dict[cls] += per_class_iou[cls]

        # 输出测试集上的结果
        avg_test_loss = total_test_loss / len(test_loader_entire)
        avg_accuracy = total_accuracy / len(test_loader_entire)
        # avg_iou = total_iou / len(test_loader_entire)

        all_class_iou = 0
        # 输出每个类别的平均 IoU
        for cls, iou in class_iou_dict.items():
            avg_class_iou = iou / len(test_loader_entire)  # 计算每个类别的平均 IoU
            print(f"类别 {cls} 的平均 IoU: {avg_class_iou}")
            all_class_iou += avg_class_iou
        avg_iou = all_class_iou / 5
        print(f"测试集上的平均损失:{avg_test_loss}")
        print(f"测试集上的平均准确率:{avg_accuracy}")
        print(f"测试集上的平均mIoU: {avg_iou}")

        # 记录到TensorBoard
        writer.add_scalar("test_loss", avg_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", avg_accuracy, total_test_step)
        writer.add_scalar("test_miou", avg_iou, i)

        # 记录每个类别的IoU到TensorBoard
        for cls, iou in class_iou_dict.items():
            writer.add_scalar(f"test_iou_class_{cls}", iou / len(test_loader_entire), total_test_step)

        total_test_step += 1

        #每10轮保存一次模型和可视化结果
        if i % 10 == 0:
            # 保存模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{i}.pth"))
            print("模型已保存")
            
            # 可视化所有测试图片并保存
            output_dir_entire = f'visualizations/entire/epoch_{i}'
            visualize_all_predictions(model, test_dataset_entire, output_dir_entire, i, device=device)
        

    writer.close()
    print("Training completed.")

