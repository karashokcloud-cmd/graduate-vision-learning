"""结果绘图模块：保存曲线、混淆矩阵、每类指标和图片案例。

输入：训练历史、混淆矩阵、测试预测、输出目录，以及是否保留图窗。
输出：四张PNG；交互模式下由train.py最后统一调用plt.show()。
注意：训练入口应先选好绘图后端，再导入本模块。
"""
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from pathlib import Path  # 从完整路径中取出图片文件名。
from pathlib import Path  # 从完整路径中取出图片文件名。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import matplotlib.pyplot as plt  # 创建、保存和关闭图窗。
import matplotlib.pyplot as plt  # 创建、保存和关闭图窗。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import numpy as np  # 生成绘图位置和处理混淆矩阵。
import numpy as np  # 生成绘图位置和处理混淆矩阵。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from PIL import Image  # 展示测试图片时读取原图。
from PIL import Image  # 展示测试图片时读取原图。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from model import CLASS_NAMES  # 图中文字与训练数字标签保持一致。
from model import CLASS_NAMES  # 图中文字与训练数字标签保持一致。

# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def save_plots(history, confusion, predictions, output_dir, show_plots=True):
def save_plots(history, confusion, predictions, output_dir, show_plots=True):
    """保存四类图；交互模式保留图窗，训练结束后统一show。"""
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。epochs = [r["epoch"] for r in history]
    epochs = [r["epoch"] for r in history]
    # 创建图窗与坐标轴；figsize以英寸计，返回fig和axes；本行：fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 开始循环：逐项取出数据；下方缩进内容重复执行。for phase in ["train", "validation"]:
    for phase in ["train", "validation"]:
        # 画出每轮损失；横轴epochs，纵轴从history取对应phase的值，marker=o显示每轮点，label用于区分训练和验证。
        axes[0].plot(epochs, [r[f"{phase}_loss"] for r in history], marker="o", label=phase)
        # 画出每轮准确率；横轴epochs，纵轴从history取对应phase的值，marker=o显示每轮点，label用于区分训练和验证。
        axes[1].plot(epochs, [r[f"{phase}_accuracy"] for r in history], marker="o", label=phase)
    # 把几个序列逐项配对，供同一条记录共同使用；本行：for ax, title in zip(axes, ["Cross-entropy loss", "Accuracy"]):
    for ax, title in zip(axes, ["Cross-entropy loss", "Accuracy"]):
        # 给当前小图设置标题；这不改变模型或数据；本行：ax.set_title(title)
        ax.set_title(title)
        # 设置横轴文字；曲线的Epoch表示已完成的训练轮次；本行：ax.set_xlabel("Epoch")
        ax.set_xlabel("Epoch")
        # 显示各条曲线或柱子的图例名称；本行：ax.legend()
        ax.legend()
        # 显示浅色网格；alpha=0.3控制透明度；本行：ax.grid(alpha=0.3)
        ax.grid(alpha=0.3)
    # 调整子图间距，避免标题、轴标签互相覆盖；rect可为总标题留空间；本行：fig.tight_layout()
    fig.tight_layout()
    # 把当前图保存为PNG；dpi控制输出分辨率；本行：fig.savefig(output_dir / "learning_curves.png", dpi=160)
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    # 判断条件；仅条件成立时执行下面缩进的分支。if not show_plots:
    if not show_plots:
        # 仅不弹窗模式下释放这张图；本行：plt.close(fig)
        plt.close(fig)

    # 创建图窗与坐标轴；figsize以英寸计，返回fig和axes；本行：fig, ax = plt.subplots(figsize=(5, 4))
    fig, ax = plt.subplots(figsize=(5, 4))
    # 把矩阵或图片绘制到坐标轴；cmap控制配色，不修改原始图像；本行：ax.imshow(confusion, cmap="Blues")
    ax.imshow(confusion, cmap="Blues")
    # 设置横轴刻度位置和标签，两者顺序必须与类别编号一致；本行：ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_xticks([0, 1], CLASS_NAMES)
    # 设置纵轴刻度位置和标签，混淆矩阵仍按真实类别排序；本行：ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    # 设置横轴文字；曲线的Epoch表示已完成的训练轮次；本行：ax.set_xlabel("Predicted class")
    ax.set_xlabel("Predicted class")
    # 设置纵轴文字；混淆矩阵行是真实类别、列是预测类别；本行：ax.set_ylabel("True class")
    ax.set_ylabel("True class")
    # 给当前小图设置标题；这不改变模型或数据；本行：ax.set_title("Test confusion matrix")
    ax.set_title("Test confusion matrix")
    # 生成整数序列；右端不包含，所以轮数写epochs+1；本行：for row in range(2):
    for row in range(2):
        # 生成整数序列；右端不包含，所以轮数写epochs+1；本行：for col in range(2):
        for col in range(2):
            # 在指定坐标写数值；ha和va设置对齐，fontsize控制字号；本行：ax.text(col, row, str(confusion[row][col]), ha="center", va="center",
            ax.text(col, row, str(confusion[row][col]), ha="center", va="center",
                    # 按当前数值或预测是否正确选择文字颜色；条件表达式为“成立取前者，否则取后者”，字号只影响显示。
                    color="white" if confusion[row][col] > np.max(confusion) / 2 else "black")
    # 调整子图间距，避免标题、轴标签互相覆盖；rect可为总标题留空间；本行：fig.tight_layout()
    fig.tight_layout()
    # 把当前图保存为PNG；dpi控制输出分辨率；本行：fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    # 判断条件；仅条件成立时执行下面缩进的分支。if not show_plots:
    if not show_plots:
        # 仅不弹窗模式下释放这张图；本行：plt.close(fig)
        plt.close(fig)


    # 图3：每类指标。precision看“判成该类的有多少是真的”；
    # recall看“真实该类有多少被找回”；F1综合两者。
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。per_class = []
    per_class = []
    # 生成整数序列；右端不包含，所以轮数写epochs+1；本行：for label in range(2):
    for label in range(2):
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。tp = int(confusion[label, label])
        tp = int(confusion[label, label])
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。predicted_count = int(confusion[:, label].sum())
        predicted_count = int(confusion[:, label].sum())
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。true_count = int(confusion[label, :].sum())
        true_count = int(confusion[label, :].sum())
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。precision = tp / predicted_count if predicted_count else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。recall = tp / true_count if true_count else 0.0
        recall = tp / true_count if true_count else 0.0
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # 把当前类的查准率、召回率与F1按顺序保存，供柱状图使用；本行：per_class.append((precision, recall, f1))
        per_class.append((precision, recall, f1))
    # 创建图窗与坐标轴；figsize以英寸计，返回fig和axes；本行：fig, ax = plt.subplots(figsize=(7, 4))
    fig, ax = plt.subplots(figsize=(7, 4))
    # 生成整数序列；右端不包含，所以轮数写epochs+1；本行：positions = np.arange(2)
    positions = np.arange(2)
    # 逐个遍历并同时得到从0开始的编号；本行：for offset, (index, name) in zip([-0.24, 0, 0.24], enumerate(["Precision", "Recall", "F1"])):
    for offset, (index, name) in zip([-0.24, 0, 0.24], enumerate(["Precision", "Recall", "F1"])):
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。bars = ax.bar(positions + offset, [values[index] for values in per_class], width=0.24, label=name)
        bars = ax.bar(positions + offset, [values[index] for values in per_class], width=0.24, label=name)
        # 开始循环：逐项取出数据；下方缩进内容重复执行。for bar in bars:
        for bar in bars:
            # 在指定坐标写数值；ha和va设置对齐，fontsize控制字号；本行：ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    # f字符串用花括号填入当前变量；:.4f为四位小数，:.1%为百分比，\n表示换行；这是记录/显示，不是计算损失。
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
    # 设置横轴刻度位置和标签，两者顺序必须与类别编号一致；本行：ax.set_xticks(positions, CLASS_NAMES)
    ax.set_xticks(positions, CLASS_NAMES)
    # 设置纵轴显示范围到1.15，为指标文字留出空间；分数本身仍在0到1；本行：ax.set_ylim(0, 1.15)
    ax.set_ylim(0, 1.15)
    # 设置纵轴文字；混淆矩阵行是真实类别、列是预测类别；本行：ax.set_ylabel("Score")
    ax.set_ylabel("Score")
    # 给当前小图设置标题；这不改变模型或数据；本行：ax.set_title("Per-class test metrics")
    ax.set_title("Per-class test metrics")
    # 显示各条曲线或柱子的图例名称；本行：ax.legend()
    ax.legend()
    # 调整子图间距，避免标题、轴标签互相覆盖；rect可为总标题留空间；本行：fig.tight_layout()
    fig.tight_layout()
    # 把当前图保存为PNG；dpi控制输出分辨率；本行：fig.savefig(output_dir / "per_class_metrics.png", dpi=160)
    fig.savefig(output_dir / "per_class_metrics.png", dpi=160)
    # 判断条件；仅条件成立时执行下面缩进的分支。if not show_plots:
    if not show_plots:
        # 仅不弹窗模式下释放这张图；本行：plt.close(fig)
        plt.close(fig)

    # 图4：直接看测试图。优先展示最多6张错误，再补正确图，总计最多9张。
    # 这是按错误优先选择的示例，不是测试集错误率的可视化估计。
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。wrong = [r for r in predictions if r[1] != r[2]]
    wrong = [r for r in predictions if r[1] != r[2]]
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。right = [r for r in predictions if r[1] == r[2]]
    right = [r for r in predictions if r[1] == r[2]]
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。examples = wrong[:6] + right[:9 - min(6, len(wrong))]
    examples = wrong[:6] + right[:9 - min(6, len(wrong))]
    # 判断条件；仅条件成立时执行下面缩进的分支。if len(examples) < 9:
    if len(examples) < 9:
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。examples += wrong[6:6 + 9 - len(examples)]
        examples += wrong[6:6 + 9 - len(examples)]
    # 创建图窗与坐标轴；figsize以英寸计，返回fig和axes；本行：fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    # 开始循环：逐项取出数据；下方缩进内容重复执行。for ax in axes.flat:
    for ax in axes.flat:
        # 隐藏图片案例的坐标轴，只保留图像和预测标题；本行：ax.axis("off")
        ax.axis("off")
    # 把几个序列逐项配对，供同一条记录共同使用；本行：for ax, (path, truth, guess, probability) in zip(axes.flat, examples):
    for ax, (path, truth, guess, probability) in zip(axes.flat, examples):
        # 进入受管理的代码块；结束时关闭文件/图片或恢复梯度开关，不需要手动善后。with Image.open(path) as image:
        with Image.open(path) as image:
            # 把矩阵或图片绘制到坐标轴；cmap控制配色，不修改原始图像；本行：ax.imshow(image.convert("RGB"))
            ax.imshow(image.convert("RGB"))
        # 给当前小图设置标题；这不改变模型或数据；本行：ax.set_title(
        ax.set_title(
            # f字符串用花括号填入当前变量；:.4f为四位小数，:.1%为百分比，\n表示换行；这是记录/显示，不是计算损失。
            f"True: {CLASS_NAMES[truth]} | Pred: {CLASS_NAMES[guess]}\n"
            # f字符串用花括号填入当前变量；:.4f为四位小数，:.1%为百分比，\n表示换行；这是记录/显示，不是计算损失。
            f"P(positive)={probability:.2f} | {Path(path).name}",
            # 按当前数值或预测是否正确选择文字颜色；条件表达式为“成立取前者，否则取后者”，字号只影响显示。
            color="darkgreen" if truth == guess else "firebrick", fontsize=9,
        # 结束上方尚未闭合的参数、列表或字典；仍属于同一条语句，不是新增训练步骤。
        )
    # 设置整组子图的总标题，说明案例按错误优先挑选；本行：fig.suptitle("Test examples: errors first (selected examples, not a random sample)", fontsize=11)
    fig.suptitle("Test examples: errors first (selected examples, not a random sample)", fontsize=11)
    # 调整子图间距，避免标题、轴标签互相覆盖；rect可为总标题留空间；本行：fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # 把当前图保存为PNG；dpi控制输出分辨率；本行：fig.savefig(output_dir / "test_examples.png", dpi=160)
    fig.savefig(output_dir / "test_examples.png", dpi=160)
    # 判断条件；仅条件成立时执行下面缩进的分支。if not show_plots:
    if not show_plots:
        # 仅不弹窗模式下释放这张图；本行：plt.close(fig)
        plt.close(fig)
