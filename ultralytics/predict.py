import sys
import argparse
import os

sys.path.append('/root/ultralytics/')  # Path 以Autodl为例

from ultralytics import YOLO

def main(opt):
    # yaml = opt.cfg
    # model = YOLO(yaml)

    model.info()

    model = YOLO('/root/ultralytics/runs/detect/train-vanillanet-2/weights/best.pt')

    # 预测图片，并保存为txt文件
    results = model.predict('/root/datasets/autodl-tmp/合并数据集/output/images/val',
                            save=True,         # 保存预测后的图片
                            save_txt=True,     # 保存预测结果为 .txt 文件
                            imgsz=640,         # 推理图像大小
                            save_conf=True         # 置信度
                        )

    # 额外的自定义操作：例如提取保存的txt文件和置信度信息
    for result in results:
        txt_file = result.path.stem + '.txt'  # 得到保存的txt文件名称
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = list(map(float, line.strip().split()))  # 将每行的内容转换为浮动数值列表
                if len(values) == 6:  # 如果行格式正确，即包含类别、位置、宽高和置信度
                    cls_id, x, y, w, h, conf = values
                    print(f"Class: {int(cls_id)}, BBox: ({x}, {y}, {w}, {h}), Confidence: {conf}")
                else:
                    print(f"Error in line format: {line.strip()}")
                    continue  # 跳过格式错误的行


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r'/root/ultralytics/runs/detect/train-vanillanet-2/weights/best.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
