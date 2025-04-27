from ultralytics import YOLO
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 模型文件路径
model_yaml_path = r"/root/ultralytics/runs/detect/train5/weights/best.pt"

# 数据集配置文件路径
data_yaml_path = r"/root/autodl-tmp/ultralytics/合并数据集.yaml"

if __name__ == "__main__":
    # 加载模型
    model = YOLO(model_yaml_path)
    
    # 模型验证
    model.val(
        data=data_yaml_path,  # 数据集路径
        split='val',          # 验证集划分
        imgsz=640,            # 输入图片大小
        batch=4,              # 批量大小
        rect=False,           # 是否启用矩形推理
        project='runs/val',   # 保存验证结果的路径
        name='exp',           # 保存的实验名称
        save_json=True        # 保存验证结果为 COCO 格式 JSON
    )
