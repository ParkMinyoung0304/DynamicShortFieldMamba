import os
from PIL import Image

def resize_images_to_ratio(input_folder, output_folder, target_size=(800, 400)):
    """
    将输入文件夹中的图片调整为目标尺寸，并保存到输出文件夹。
    :param input_folder: 原始图像所在文件夹
    :param output_folder: 调整后图像保存的文件夹
    :param target_size: 目标尺寸，宽:高 = 2:1, 默认800x400
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):  # 只处理图片
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # 调整图像大小，使用 LANCZOS 替代 ANTIALIAS
            resized_img = img.resize(target_size, Image.LANCZOS)
            
            # 保存调整后的图像
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)

            print(f"已处理: {filename}，保存到: {output_path}")

if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = "model/visual"  # 替换为包含原始图片的文件夹路径
    output_folder = "model/visual_resize"  # 替换为保存调整后图片的文件夹路径
    
    # 目标尺寸，宽:高 = 2:1，例如800x400
    target_size = (400, 800)

    # 运行函数，将图片resize成2:1
    resize_images_to_ratio(input_folder, output_folder, target_size)
