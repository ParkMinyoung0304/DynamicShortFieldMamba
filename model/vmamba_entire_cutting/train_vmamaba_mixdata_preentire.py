import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from vmamba_unet import VMambaUnet  # Assuming VMambaUnet is suitable for segmentation
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import random
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
from color_map_converted import COLOR_MAP 
import numpy as np


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
    return DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2), DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

def visualize_augmentations(dataset, indices=[0, 1, 2], num_samples=3):
    print("开始图像增强可视化")
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 10))
    for i, idx in enumerate(indices[:num_samples]):
        original_img, original_mask = dataset[idx]
        transformed_img, transformed_mask = dataset.transform(original_img.clone(), original_mask.clone())
        
        # 应用逆归一化
        original_img = unnormalize(original_img).permute(1, 2, 0)
        transformed_img = unnormalize(transformed_img).permute(1, 2, 0)

        # 显示原始图像
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f'Original Image {idx}')
        axes[i, 0].axis('off')

        # 显示原始掩码
        original_mask = original_mask.squeeze()  # 移除额外的单一维度
        axes[i, 1].imshow(original_mask, cmap='gray')
        axes[i, 1].set_title(f'Original Mask {idx}')
        axes[i, 1].axis('off')

        # 显示变换后的图像
        axes[i, 2].imshow(transformed_img)
        axes[i, 2].set_title(f'Transformed Image {idx}')
        axes[i, 2].axis('off')

        # 显示变换后的掩码
        transformed_mask = transformed_mask.squeeze()  # 移除额外的单一维度
        axes[i, 3].imshow(transformed_mask, cmap='gray')
        axes[i, 3].set_title(f'Transformed Mask {idx}')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('model/visual/visualizations.png')  # 保存图像到文件
    plt.close()
    print("结束图像增强可视化")

def visualize_predictions_entire_images(model, dataset, indices=[0, 1, 2], num_samples=3, device='cpu'):
    print("开始预测可视化")
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 10))
    num_classes = len(COLOR_MAP) + 1

    with torch.no_grad():
        for i, idx in enumerate(indices[:num_samples]):
            image, true_mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备
            true_mask = true_mask.cpu().numpy()
            pred_mask = model(image).argmax(1).squeeze(0).cpu().numpy()  # 获取预测掩码
            image = unnormalize(image.squeeze(0)).permute(1, 2, 0).cpu().numpy()

            # 打印预测掩码的唯一值
            # print(f"Image {idx} - Predicted Mask unique values: {np.unique(pred_mask)}")
            # print(f"Image {idx} - True Mask unique values: {np.unique(true_mask)}")

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {idx}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask, cmap='gray', vmin=0, vmax=num_classes-1)
            axes[i, 1].set_title(f'True Mask {idx}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=num_classes-1)
            axes[i, 2].set_title(f'Predicted Mask {idx}')
            axes[i, 2].axis('off')

            # Ensure both true_mask and pred_mask are of the same dtype and range
            true_mask = true_mask.astype(np.int32)
            pred_mask = pred_mask.astype(np.int32)

            diff = (true_mask != pred_mask).astype(np.uint8)  # 计算差异掩码
            axes[i, 3].imshow(diff, cmap='gray')
            axes[i, 3].set_title(f'Difference {idx}')
            axes[i, 3].axis('off')
    print("结束预测可视化")
    plt.tight_layout()
    plt.savefig('model/visual/predictions_entire_images_epoch_{}.png'.format(num_epochs))
    plt.close()

def visualize_predictions_all(model, dataset, indices=[0, 1, 2], num_samples=3, device='cpu'):
    print("开始预测可视化")
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 10))
    num_classes = len(COLOR_MAP) + 1

    with torch.no_grad():
        for i, idx in enumerate(indices[:num_samples]):
            image, true_mask = dataset[idx]
            image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备
            true_mask = true_mask.cpu().numpy()
            pred_mask = model(image).argmax(1).squeeze(0).cpu().numpy()  # 获取预测掩码
            image = unnormalize(image.squeeze(0)).permute(1, 2, 0).cpu().numpy()

            # 打印预测掩码的唯一值
            # print(f"Image {idx} - Predicted Mask unique values: {np.unique(pred_mask)}")
            # print(f"Image {idx} - True Mask unique values: {np.unique(true_mask)}")

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {idx}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask, cmap='gray', vmin=0, vmax=num_classes-1)
            axes[i, 1].set_title(f'True Mask {idx}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=num_classes-1)
            axes[i, 2].set_title(f'Predicted Mask {idx}')
            axes[i, 2].axis('off')

            # Ensure both true_mask and pred_mask are of the same dtype and range
            true_mask = true_mask.astype(np.int32)
            pred_mask = pred_mask.astype(np.int32)

            diff = (true_mask != pred_mask).astype(np.uint8)  # 计算差异掩码
            axes[i, 3].imshow(diff, cmap='gray')
            axes[i, 3].set_title(f'Difference {idx}')
            axes[i, 3].axis('off')
    print("结束预测可视化")
    plt.tight_layout()
    plt.savefig('model/visual/predictions_all_epoch_{}.png'.format(num_epochs))
    plt.close()

def calculate_miou(pred, target, num_classes):
    """计算mIoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()  # 计算交集
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection  # 计算并集
        if union == 0:
            iou = float('nan')  # 避免除以零
        else:
            iou = intersection / union
        ious.append(iou)

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else float('nan')  # 计算有效IoU的平均值
    return miou

transform = Transform()
# Load datasets
train_loader_all, test_loader_all = load_dataset('data/mix/images', 'data/mix/masks', transform)
train_loader_entire, test_loader_entire = load_dataset('data/entire_img/images', 'data/entire_img/masks', transform)


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
        visualize_augmentations(train_loader_all.dataset, indices=[0, 1, 2], num_samples=3)
        visualize_augmentations(train_loader_entire.dataset, indices=[0, 1, 2], num_samples=3)
        train_loader, test_loader = train_loader_all, test_loader_all

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
        
        # print("模型输出的最大值:", outputs.max().item())
        # print("模型输出的最小值:", outputs.min().item())

        # 测试步骤
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        total_iou = 0
        with torch.no_grad():
            for images, labels in test_loader_entire:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                total_test_loss += loss.item()

                predictions = outputs.argmax(1)
                accuracy = (predictions == labels).float().mean().item()
                total_accuracy += accuracy

                miou = calculate_miou(predictions, labels, num_classes=len(COLOR_MAP))
                total_iou += miou  # 将当前批次的mIoU累加

        avg_test_loss = total_test_loss / len(test_loader_entire)
        avg_accuracy = total_accuracy / len(test_loader_entire)
        avg_iou = total_iou / len(test_loader_entire)

        print(f"测试集上的平均损失:{avg_test_loss}")
        print(f"测试集上的平均准确率:{avg_accuracy}")
        print(f"测试集上的平均mIoU: {avg_iou}")

        writer.add_scalar("test_loss", avg_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", avg_accuracy, total_test_step)
        writer.add_scalar("test_miou", avg_iou, i)  # 将平均mIoU记录到TensorBoard
        total_test_step += 1

        #每10轮保存一次模型
        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{i}.pth"))
        print("模型已保存")

        visualize_predictions_entire_images(model, test_loader_entire.dataset, indices=[0, 1, 2], num_samples=3, device=device)
        visualize_predictions_all(model, test_loader_all.dataset, indices=[0, 1, 2], num_samples=3, device=device)
        

    writer.close()
    print("Training completed.")

