




from tidecv import TIDE
import tidecv.datasets
import matplotlib.pyplot as plt
import os

# 配置中文支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体（你可以选择其他字体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# json文件路径
annFile = '/root/yolo-coco.json'  # 数据集标注json文件路径
resFile = '/root/runs/val/exp2/predictions.json'  # yolo系列val保存的json结果文件路径

# 加载 COCO 格式的标注数据和预测结果
gt = tidecv.datasets.COCO(annFile)
bbox_results = tidecv.datasets.COCOResult(resFile)

# 初始化 TIDE 工具
tide = TIDE()

# 使用 TIDE 进行评估
tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX, name='Detection Evaluation')

# 打印评估摘要
tide.summarize()

# 指定输出文件夹路径
output_folder = '/root/tide_results/'
os.makedirs(output_folder, exist_ok=True)  # 创建文件夹，如果文件夹已存在则不会抛出错误

# 添加分母为零时的处理，避免ZeroDivisionError
try:
    # 保存评估图表到文件夹
    tide.plot(out_dir=output_folder)
    print(f"Evaluation results saved to {output_folder}")
except ZeroDivisionError:
    print("Error during result plotting: Division by zero encountered.")
    # 如果分母为零，可以选择忽略或返回默认值




