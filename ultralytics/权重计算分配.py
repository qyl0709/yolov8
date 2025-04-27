import os
import glob
import json
import numpy as np
from collections import Counter
import yaml


def calculate_class_weights(label_dir, num_classes):
    """
    统计每个类别的样本数量并计算类别权重。
    
    :param label_dir: 标注文件夹路径（假设每个标注文件对应一个图像）
    :param num_classes: 类别总数
    :return: 每个类别的权重列表
    """
    class_counts = Counter()
    
    # 获取所有标注文件
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))  # 假设是YOLO格式的txt文件
    
    for label_file in label_files:
        with open(label_file, "r") as f:
            # 读取标注文件的每一行，提取类别ID
            for line in f:
                class_id = int(line.strip().split()[0])  # 获取类别ID
                class_counts[class_id] += 1
    
    # 计算每个类别的权重
    total_samples = sum(class_counts.values())  # 总样本数
    class_weights = []
    
    for i in range(num_classes):
        class_freq = class_counts.get(i, 0)  # 获取该类别的样本数量，若没有则为0
        # 使用反比的方式计算类别权重
        weight = total_samples / (num_classes * (class_freq + 1e-6))  # 防止除零错误
        class_weights.append(weight)
    
    # 可以归一化一下权重，使得总权重不太大
    class_weights = np.array(class_weights)
    class_weights /= class_weights.max()  # 最大值归一化
    
    return class_weights.tolist()

def main():
    label_dir = "/root/datasets/autodl-tmp/合并数据集/output/labels/train"  # 替换成你的标注文件夹路径
    num_classes = 124 # 替换成你的类别数量
    
    # 计算类别权重
    class_weights = calculate_class_weights(label_dir, num_classes)
    
    # 打印类别权重
    print("Calculated class weights:", class_weights)
    
    # 如果你要生成配置文件，可以这样写
    config = {
        "loss": {
            "name": "FocalLoss",  # 损失函数
            "params": {
                "alpha": 0.75,  # Focal系数
                "gamma": 1.5,   # 困难样本权重
                "class_weight": class_weights  # 使用计算的类别权重
            }
        }
    }
    
    # 保存为 YAML 配置文件
    with open("权重计算分配.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Configuration has been saved to config.yaml")

if __name__ == "__main__":
    main()
