import os
from PIL import Image
from tqdm import tqdm

def calculate_average_aspect_ratio(image_folder):
    aspect_ratios = []
    for filename in tqdm(os.listdir(image_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 支持的图像格式
            file_path = os.path.join(image_folder, filename)
            with Image.open(file_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                aspect_ratios.append(aspect_ratio)

    if aspect_ratios:
        average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
        return average_aspect_ratio
    else:
        return None

def recommend_dimensions(average_aspect_ratio, base_height=256):
    """基于基础高度和平均宽高比推荐新的尺寸"""
    recommended_width = int(average_aspect_ratio * base_height)
    return (recommended_width, base_height)

# 使用示例
image_folder = 'data/cutting/images'  # 替换为您的图像文件夹路径
average_aspect_ratio = calculate_average_aspect_ratio(image_folder)
if average_aspect_ratio:
    recommended_size = recommend_dimensions(average_aspect_ratio)
    print(f"Recommended image dimensions: {recommended_size}")
else:
    print("No images found or unsupported image format.")
