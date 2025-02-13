import re

def read_classes_and_colors(file_path):
    """从文本文件中读取颜色和类别数据，并生成列表。"""
    classes_and_colors = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(", Color: ")
            class_part = parts[0]
            color_part = parts[1]
            class_name = class_part.split(": ")[1]
            color = tuple(eval(color_part))
            classes_and_colors.append((class_name, color))
    return classes_and_colors

def generate_color_map_and_save(file_path, classes_and_colors):
    """生成颜色映射字典，并将结果保存到文本文件中。"""
    with open(file_path, 'w') as file:
        file.write("COLOR_MAP = {\n")
        for index, (class_name, color) in enumerate(classes_and_colors):
            if index < len(classes_and_colors) - 1:
                line = f"    {color}: {index},  # {class_name}\n"
            else:
                line = f"    {color}: {index}  # {class_name}\n"
            file.write(line)
        file.write("}\n")

# 使用这些函数
input_file_path = "data/color_map.txt"  # 输入文件的路径
output_file_path = "data/color_map_converted.py"  # 输出文件的路径

classes_and_colors = read_classes_and_colors(input_file_path)
generate_color_map_and_save(output_file_path, classes_and_colors)



