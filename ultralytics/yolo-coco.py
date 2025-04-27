


import os
import json
from PIL import Image

# 配置路径
images_dir = r'/root/datasets/autodl-tmp/合并数据集/output/images/val'  # 图片文件夹路径
labels_dir = r'/root/datasets/autodl-tmp/合并数据集/output/labels/val'  # 标注文件夹路径
classes_path = r'/root/datasets/autodl-tmp/合并数据集/classes.txt'  # 类别文件路径，每行一个类别
output_json_path = '/root/yolo-coco.json'  # 转换后保存的 JSON 文件路径

# 从 classes.txt 文件中读取类别信息，动态生成 COCO 的 categories 字段
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]  # 每行是一个类别名称

# 自动生成 COCO 格式的 categories 字段
categories = [{"id": idx + 1, "name": cls_name} for idx, cls_name in enumerate(classes)]

# 初始化 COCO 数据集字典结构
coco_data = {
    "images": [],
    "annotations": [],
    "categories": categories  # 动态生成的类别信息
}

def yolo_to_coco(images_dir, labels_dir, output_path):
    anno_id = 1  # 初始化注释 ID

    for label_file in os.listdir(labels_dir):
        # 分割文件名和扩展名，获取图像 ID
        image_id, _ = os.path.splitext(label_file)

        # 构建图像路径，获取图像尺寸
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue  # 如果图片文件不存在，跳过

        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息到 COCO 数据集
        coco_data["images"].append({
            "file_name": f"{image_id}.jpg",
            "height": height,
            "width": width,
            "id": int(''.join(filter(str.isdigit, image_id)))  # 提取数字部分并转为整型
        })

        # 打开相应的标注文件
        with open(os.path.join(labels_dir, label_file), 'r') as file:
            for line in file:
                # 解析 YOLO 的标注格式
                category_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

                # 将 YOLO 坐标转换成 COCO 坐标
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                width_bbox = bbox_width * width
                height_bbox = bbox_height * height

                # 添加注释信息到 COCO 数据集
                coco_data["annotations"].append({
                    "id": anno_id,
                    "image_id": int(''.join(filter(str.isdigit, image_id))),  # 提取数字部分并转为整型
                    "category_id": int(category_id) + 1,  # 类别 ID 从 1 开始
                    "bbox": [x_min, y_min, width_bbox, height_bbox],  # 边界框
                    "area": width_bbox * height_bbox,  # 面积
                    "iscrowd": 0,  # 是否为复杂目标
                    "segmentation": []  # 空的分割信息，检测任务中可以为空
                })
                anno_id += 1

    # 写入 COCO 格式的 JSON 文件
    with open(output_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

# 调用函数进行转换
yolo_to_coco(images_dir, labels_dir, output_json_path)









# import os
# import json
# from PIL import Image

# # 配置路径
# images_dir = r'/root/datasets/autodl-tmp/合并数据集/output/images/val'  # 图片文件夹路径
# labels_dir = r'/root/datasets/autodl-tmp/合并数据集/output/labels/val'  # 标注文件夹路径
# classes_path = r'/root/datasets/autodl-tmp/合并数据集/classes.txt'  # 类别文件路径，每行一个类别
# output_json_path = '/root/yolo-coco.json'  # 转换后保存的 JSON 文件路径

# # 从 classes.txt 文件中读取类别信息，动态生成 COCO 的 categories 字段
# with open(classes_path, 'r') as f:
#     classes = [line.strip() for line in f.readlines()]  # 每行是一个类别名称

# # 自动生成 COCO 格式的 categories 字段
# categories = [{"id": idx + 1, "name": cls_name} for idx, cls_name in enumerate(classes)]

# # 初始化 COCO 数据集字典结构
# coco_data = {
#     "images": [],
#     "annotations": [],
#     "categories": categories  # 动态生成的类别信息
# }

# def yolo_to_coco(images_dir, labels_dir, output_path):
#     anno_id = 1  # 初始化注释 ID

#     for label_file in os.listdir(labels_dir):
#         # 分割文件名和扩展名，获取图像 ID
#         image_id, _ = os.path.splitext(label_file)

#         # 构建图像路径，获取图像尺寸
#         image_path = os.path.join(images_dir, f"{image_id}.jpg")
#         if not os.path.exists(image_path):
#             print(f"Warning: Image {image_path} not found, skipping...")
#             continue  # 如果图片文件不存在，跳过

#         with Image.open(image_path) as img:
#             width, height = img.size

#         # 添加图像信息到 COCO 数据集
#         coco_data["images"].append({
#             "file_name": f"{image_id}.jpg",
#             "height": height,
#             "width": width,
#             "id": int(''.join(filter(str.isdigit, image_id)))  # 提取数字部分并转为整型
#         })

#         # 打开相应的标注文件
#         with open(os.path.join(labels_dir, label_file), 'r') as file:
#             for line in file:
#                 # 解析 YOLO 的标注格式
#                 category_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

#                 # 将 YOLO 坐标转换成 COCO 坐标（归一化到[0, 1]范围）
#                 x_center_norm = round(x_center / width, 3)
#                 y_center_norm = round(y_center / height, 3)
#                 bbox_width_norm = round(bbox_width / width, 3)
#                 bbox_height_norm = round(bbox_height / height, 3)

#                 # 添加注释信息到 COCO 数据集
#                 coco_data["annotations"].append({
#                     "id": anno_id,
#                     "image_id": int(''.join(filter(str.isdigit, image_id))),  # 提取数字部分并转为整型
#                     "category_id": int(category_id) + 1,  # 类别 ID 从 1 开始
#                     "bbox": [
#                         round(x_center_norm - bbox_width_norm / 2, 3),  # 转换成 COCO 格式，并保留三位小数
#                         round(y_center_norm - bbox_height_norm / 2, 3),
#                         bbox_width_norm,
#                         bbox_height_norm
#                     ],
#                     "area": round(bbox_width_norm * bbox_height_norm, 3),  # 面积
#                     "iscrowd": 0,  # 是否为复杂目标
#                     "segmentation": []  # 空的分割信息，检测任务中可以为空
#                 })
#                 anno_id += 1

#     # 写入 COCO 格式的 JSON 文件
#     with open(output_path, 'w') as json_file:
#         json.dump(coco_data, json_file, indent=4)

# # 调用函数进行转换
# yolo_to_coco(images_dir, labels_dir, output_json_path)











