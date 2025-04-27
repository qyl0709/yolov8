import sys
import argparse
import os



# sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    
    
    # model = YOLO("/root/ultralytics/runs/detect/train3/weights/last.pt")
    

    model.info()

    results = model.train(data='合并数据集.yaml',      # 数据集路径
        epochs=150,                # 增加训练轮数
        imgsz=640,                 # 输入图像大小
        workers=8,                 # 多线程数据加载
        batch=64,                  # 增加批量大小
        # optimizer='Adam',          # 使用 Adam 优化器（默认 SGD）
        # lr0=1e-3,                  # 初始学习率
        # momentum=0.937,            # 动量
        # weight_decay=5e-4,         # 权重衰减
        # patience=10,               # 提前停止：监控验证指标
        # augment=True,              # 启用数据增强
        # val=True,                  # 启用验证
        # device=0,                  # 使用 GPU（设备 0）         
        # resume=True,  
        
        # cls_loss='focal',      # 使用 Focal Loss 作为分类损失
        # box_loss='ciou',       # 使用 CIoU Loss 作为回归损失
                        )
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/混合.yaml', help='initial weights path')#ultralytics/cfg/models/v8/yolov8.yaml
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)