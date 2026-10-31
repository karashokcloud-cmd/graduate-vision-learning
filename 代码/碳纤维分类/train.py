"""运行入口：读配置，组织数据、模型、训练、评估和绘图。
命令仍为python train.py；--help只显示用法，不训练。
"""
# ==================== 1. 导入工具 ====================
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import argparse
import argparse
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import csv
import csv
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import json
import json
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import random
import random
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import sys
import sys
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from datetime import datetime
from datetime import datetime
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from pathlib import Path
from pathlib import Path

# 捕获依赖导入错误并给出清晰提示；这不是安装命令。
try:
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import matplotlib
    import matplotlib
    
    # 有桌面的Windows电脑默认显示图窗；--no-show-plots仅保存。
    # 在导入pyplot之前选择绘图后端，避免Agg后端无法弹窗。
    # 在导入pyplot前选择后端；Agg仅保存，TkAgg允许桌面弹窗；本行：matplotlib.use("Agg" if "--no-show-plots" in sys.argv else "TkAgg")
    matplotlib.use("Agg" if "--no-show-plots" in sys.argv else "TkAgg")
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import numpy as np
    import numpy as np
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import torch
    import torch
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from PIL import Image
    from PIL import Image
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from torch import nn
    from torch import nn
    # 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from torch.utils.data import DataLoader
    from torch.utils.data import DataLoader
# 捕获依赖导入错误并给出清晰提示；这不是安装命令。
except ImportError as exc:
    # 条件不满足时主动停止并报告原因，避免拿错误数据继续得到假结果。raise SystemExit(
    raise SystemExit(
        # 组成依赖导入失败时给用户的说明文字；不会自动执行安装。
        "依赖尚未装好。请按002教程安装 PyTorch 和 requirements.txt。\n"
        # f字符串用花括号填入当前变量；:.4f为四位小数，:.1%为百分比，\n表示换行；这是记录/显示，不是计算损失。
        f"当前解释器：{sys.executable}\n原始错误：{exc}"
    # 结束SystemExit说明并保留原始异常的因果关系，便于定位究竟缺少哪个依赖。
    ) from exc

# 下面四个模块都属于本工程，放在train.py旁边即可导入。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from data import prepare_splits, CarbonDataset  # 数据划分与逐张读图。
from data import prepare_splits, CarbonDataset  # 数据划分与逐张读图。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from model import CLASS_NAMES, WEIGHTS, build_model  # 标签定义与模型。
from model import CLASS_NAMES, WEIGHTS, build_model  # 标签定义与模型。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from engine import run_one_phase  # 训练、验证和测试共用的计算步骤。
from engine import run_one_phase  # 训练、验证和测试共用的计算步骤。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from plots import save_plots  # 四种结果图的生成。
from plots import save_plots  # 四种结果图的生成。

# __file__是本源码的位置；配置与输出相对源码放置，不依赖终端当前目录。
# 当前train.py所在目录，用它定位配置、权重缓存和结果；与终端当前目录无关；本行赋值/累计：PROJECT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parent
# 是否在保存后保留并显示图窗；本行赋值/累计：SHOW_PLOTS = "--no-show-plots" not in sys.argv
SHOW_PLOTS = "--no-show-plots" not in sys.argv

# 随机种子，控制划分、初始化等随机过程；本行赋值/累计：SEED = 20261029
SEED = 20261029
# 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。EPOCHS = 30                # 上限来自config.json，可用--epochs临时覆盖。
EPOCHS = 30                # 上限来自config.json，可用--epochs临时覆盖。
# 一个批次的图片数；本行赋值/累计：BATCH_SIZE = 16            # 一次处理16张图片，最后一个批次可以少于16张。
BATCH_SIZE = 16            # 一次处理16张图片，最后一个批次可以少于16张。
# 优化器学习率，不是模型权重；本行赋值/累计：LEARNING_RATE = 0.001      # 更新分类头参数的步长。
LEARNING_RATE = 0.001      # 更新分类头参数的步长。

# 让预训练权重缓存也保存在D盘工程，而不是默认的C盘用户目录。
# 设置官方预训练权重的本机缓存目录；这里只指定位置，不立即下载；本行：torch.hub.set_dir(str(PROJECT / "weights"))
torch.hub.set_dir(str(PROJECT / "weights"))


# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def set_seed():
def set_seed():
    """固定随机种子，方便重跑比较；不保证不同硬件/版本逐位完全一致。"""
    # 固定Python随机序列，帮助复现划分和其他随机操作；本行：random.seed(SEED)
    random.seed(SEED)
    # 固定Python随机序列，帮助复现划分和其他随机操作；本行：np.random.seed(SEED)
    np.random.seed(SEED)
    # 固定PyTorch初始化等随机操作，使同条件比较更稳定；本行：torch.manual_seed(SEED)
    torch.manual_seed(SEED)
    # 限定CPU计算线程数为4，不是批大小也不是训练轮数；本行：torch.set_num_threads(4)  # 第一课使用CPU，避免同时开过多线程。
    torch.set_num_threads(4)  # 第一课使用CPU，避免同时开过多线程。


# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def train(epochs):
def train(epochs):
    """一次完整流程：读数据→划分→加载模型→训练/验证→最终测试→保存。"""
    # 把JSON文本解析为Python字典；本行：config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8-sig"))
    config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8-sig"))
    # 声明这个函数会更新模块级设置，后续读取同一设置；这些是超参数，不是网络权重。
    global SEED, BATCH_SIZE, LEARNING_RATE
    # 随机种子，控制划分、初始化等随机过程；本行赋值/累计：SEED = int(config["seed"])
    SEED = int(config["seed"])
    # 一个批次的图片数；本行赋值/累计：BATCH_SIZE = int(config["batch_size"])
    BATCH_SIZE = int(config["batch_size"])
    # 优化器学习率，不是模型权重；本行赋值/累计：LEARNING_RATE = float(config["learning_rate"])
    LEARNING_RATE = float(config["learning_rate"])
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。patience = int(config["early_stopping_patience"])
    patience = int(config["early_stopping_patience"])
    # 判断条件；仅条件成立时执行下面缩进的分支。if BATCH_SIZE < 1 or LEARNING_RATE <= 0 or patience < 1:
    if BATCH_SIZE < 1 or LEARNING_RATE <= 0 or patience < 1:
        # 条件不满足时主动停止并报告原因，避免拿错误数据继续得到假结果。raise ValueError("batch_size、learning_rate和early_stopping_patience必须为正数。")
        raise ValueError("batch_size、learning_rate和early_stopping_patience必须为正数。")
    # 执行本文件上方的随机种子设置函数，不启动训练；本行：set_seed()
    set_seed()
    # 配置中指定的原始图片目录，只读原图；本行赋值/累计：data_dir = Path(config["binary_data"]).expanduser()
    data_dir = Path(config["binary_data"]).expanduser()
    # 判断条件；仅条件成立时执行下面缩进的分支。if not data_dir.is_absolute():
    if not data_dir.is_absolute():
        # 配置中指定的原始图片目录，只读原图；本行赋值/累计：data_dir = PROJECT / data_dir
        data_dir = PROJECT / data_dir
    # 判断条件；仅条件成立时执行下面缩进的分支。if not data_dir.is_dir():
    if not data_dir.is_dir():
        # 条件不满足时主动停止并报告原因，避免拿错误数据继续得到假结果。raise FileNotFoundError(f"数据目录不存在：{data_dir}；请修改config.json中的binary_data。")
        raise FileNotFoundError(f"数据目录不存在：{data_dir}；请修改config.json中的binary_data。")
    # 读取目录并用固定seed划分，不靠文件名猜测缺陷位置；本行：splits = prepare_splits(data_dir, seed=SEED)
    splits = prepare_splits(data_dir, seed=SEED)
    # 每次新建结果目录，不覆盖以前的实验。
    # 这次运行独有的时间戳目录，不覆盖上次结果；本行赋值/累计：output_dir = PROJECT / "runs" / "resnet18" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT / "runs" / "resnet18" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # 创建本次结果目录；parents=True补齐父目录；exist_ok=False避免覆盖已有同名实验；本行：output_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    # 保存实际使用的配置；命令行覆盖轮数时也能追溯。
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。actual_config = dict(config, max_epochs=epochs, binary_data=str(data_dir), show_plots=SHOW_PLOTS)
    actual_config = dict(config, max_epochs=epochs, binary_data=str(data_dir), show_plots=SHOW_PLOTS)
    # 把Python结构编码为JSON文本；indent是缩进，ensure_ascii=False保留中文；本行：(output_dir / "config_used.json").write_text(json.dumps(actual_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config_used.json").write_text(json.dumps(actual_config, ensure_ascii=False, indent=2), encoding="utf-8")
    # 进入受管理的代码块；结束时关闭文件/图片或恢复梯度开关，不需要手动善后。with (output_dir / "split.csv").open("w", encoding="utf-8-sig", newline="") as f:
    with (output_dir / "split.csv").open("w", encoding="utf-8-sig", newline="") as f:
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。writer = csv.writer(f)
        writer = csv.writer(f)
        # 写一行CSV；列表各项分别成为一列；本行：writer.writerow(["path", "label", "class_name", "split"])
        writer.writerow(["path", "label", "class_name", "split"])
        # 开始循环：逐项取出数据；下方缩进内容重复执行。for phase, records in splits.items():
        for phase, records in splits.items():
            # 把多条记录逐行写入CSV；本行：writer.writerows((str(p), label, CLASS_NAMES[label], phase) for p, label in records)
            writer.writerows((str(p), label, CLASS_NAMES[label], phase) for p, label in records)

    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。loaders = {
    loaders = {
        # 把Dataset组装成批次；只训练时shuffle；num_workers=0避免Windows子进程问题；本行：phase: DataLoader(CarbonDataset(records), batch_size=BATCH_SIZE,
        phase: DataLoader(CarbonDataset(records), batch_size=BATCH_SIZE,
                          # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。shuffle=(phase == "train"), num_workers=0)
                          shuffle=(phase == "train"), num_workers=0)
        # 开始循环：逐项取出数据；下方缩进内容重复执行。for phase, records in splits.items()
        for phase, records in splits.items()
    # 结束上方尚未闭合的参数、列表或字典；仍属于同一条语句，不是新增训练步骤。
    }
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。device = torch.device("cpu")  # 第一遍先用CPU跑通，不依赖显卡和CUDA配置。
    device = torch.device("cpu")  # 第一遍先用CPU跑通，不依赖显卡和CUDA配置。
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"Python：{sys.executable}", flush=True)
    print(f"Python：{sys.executable}", flush=True)
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"数据：{data_dir}\n划分：280训练 / 40验证 / 80测试", flush=True)
    print(f"数据：{data_dir}\n划分：280训练 / 40验证 / 80测试", flush=True)
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"计算设备：{device}，结果目录：{output_dir}", flush=True)
    print(f"计算设备：{device}，结果目录：{output_dir}", flush=True)
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print("正在加载ImageNet预训练权重，第一次运行需要联网下载……", flush=True)
    print("正在加载ImageNet预训练权重，第一次运行需要联网下载……", flush=True)
    # 将图片、标签或模型放到指定计算设备；当前device是CPU，双方设备需要一致。
    model = build_model(use_pretrained=True).to(device)
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print("仅训练新的全连接层，参数数目：",
    print("仅训练新的全连接层，参数数目：",
          # 只对requires_grad=True的参数逐个数元素并求和，得到1026；flush=True让终端及时显示。
          sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    # 创建交叉熵；输入logits和数字标签，内部处理log-softmax；本行：criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # Adam(params,lr)：params指定只能更新分类层；lr来自配置；本行：optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    # 逐轮保存训练和验证指标的列表；本行赋值/累计：history, best_loss, best_epoch = [], float("inf"), 0
    history, best_loss, best_epoch = [], float("inf"), 0

    # 连续没有改善的验证轮数；本行赋值/累计：stale_epochs = 0
    stale_epochs = 0
    # 训练期间不评估测试集；只根据验证损失选择最佳轮次。
    # 生成整数序列；右端不包含，所以轮数写epochs+1；本行：for epoch in range(1, epochs + 1):
    for epoch in range(1, epochs + 1):
        # optimizer存在就训练；省略则只评估；返回损失、准确率、逐图输出；本行：train_loss, train_acc, _ = run_one_phase(model, loaders["train"], criterion, device, optimizer)
        train_loss, train_acc, _ = run_one_phase(model, loaders["train"], criterion, device, optimizer)
        # optimizer存在就训练；省略则只评估；返回损失、准确率、逐图输出；本行：val_loss, val_acc, _ = run_one_phase(model, loaders["validation"], criterion, device)
        val_loss, val_acc, _ = run_one_phase(model, loaders["validation"], criterion, device)
        # 把当前轮的训练和验证损失、准确率加入列表，之后画曲线；本行：history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        # 字典中的字段：冒号前是保存/查找时用的键，后面是本次字段的值。"validation_loss": val_loss, "validation_accuracy": val_acc})
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"Epoch {epoch}/{epochs} | train loss={train_loss:.4f}, acc={train_acc:.1%}"
        print(f"Epoch {epoch}/{epochs} | train loss={train_loss:.4f}, acc={train_acc:.1%}"
              # f字符串用花括号填入当前变量；:.4f为四位小数，:.1%为百分比，\n表示换行；这是记录/显示，不是计算损失。
              f" | validation loss={val_loss:.4f}, acc={val_acc:.1%}", flush=True)
        # 判断条件；仅条件成立时执行下面缩进的分支。if val_loss < best_loss:
        if val_loss < best_loss:
            # 目前最小验证损失，用来决定是否保存模型；本行赋值/累计：best_loss, best_epoch = val_loss, epoch
            best_loss, best_epoch = val_loss, epoch
            # 连续没有改善的验证轮数；本行赋值/累计：stale_epochs = 0
            stale_epochs = 0
            # 把参数字典存到本次结果目录，不是上传网络；本行：torch.save({
            torch.save({
                # 从梯度计算图分离结果，再移到CPU或转列表用于记录；不改变训练所需的损失。
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                # 字典中的字段：冒号前是保存/查找时用的键，后面是本次字段的值。"class_names": CLASS_NAMES, "weights_name": "IMAGENET1K_V1", "epoch": epoch,
                "class_names": CLASS_NAMES, "weights_name": "IMAGENET1K_V1", "epoch": epoch,
            # 结束待保存的参数字典；路径指向本次结果目录的best_model.pt，不覆盖其他实验。
            }, output_dir / "best_model.pt")
        # 当前面条件不成立时执行这个分支；具体目的看对应if。
        else:
            # 连续没有改善的验证轮数；本行赋值/累计：stale_epochs += 1
            stale_epochs += 1
        # 进入受管理的代码块；结束时关闭文件/图片或恢复梯度开关，不需要手动善后。with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            # 按DictWriter预先指定的字段顺序写CSV表头；本行：writer.writeheader()
            writer.writeheader()
            # 把多条记录逐行写入CSV；本行：writer.writerows(history)
            writer.writerows(history)
        # 验证集损失连续多轮没有降低就结束；不看测试集决定何时停。
        # 判断条件；仅条件成立时执行下面缩进的分支。if stale_epochs >= patience:
        if stale_epochs >= patience:
            # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"验证损失连续{patience}轮没有改善，提前停止。", flush=True)
            print(f"验证损失连续{patience}轮没有改善，提前停止。", flush=True)
            # 跳出训练轮次循环；随后仍会载入最佳模型、做最终测试并保存图。
            break

    # 训练结束，重新装入验证集选中的参数，再对80张测试图评估一次。
    # 读取自己保存的模型；map_location指定CPU，weights_only=True限制反序列化内容；本行：checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    # 把保存的参数装回当前模型；此后评估使用最佳轮次的参数；本行：model.load_state_dict(checkpoint["model_state"])
    model.load_state_dict(checkpoint["model_state"])
    # optimizer存在就训练；省略则只评估；返回损失、准确率、逐图输出；本行：test_loss, test_acc, predictions = run_one_phase(model, loaders["test"], criterion, device)
    test_loss, test_acc, predictions = run_one_phase(model, loaders["test"], criterion, device)
    # 行是真实类别、列是预测类别的2×2计数矩阵；本行赋值/累计：confusion = np.zeros((2, 2), dtype=int)
    confusion = np.zeros((2, 2), dtype=int)
    # 开始循环：逐项取出数据；下方缩进内容重复执行。for _, truth, guess, _ in predictions:
    for _, truth, guess, _ in predictions:
        # 行是真实类别、列是预测类别的2×2计数矩阵；本行赋值/累计：confusion[truth, guess] += 1
        confusion[truth, guess] += 1
    # 进入受管理的代码块；结束时关闭文件/图片或恢复梯度开关，不需要手动善后。with (output_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。writer = csv.writer(f)
        writer = csv.writer(f)
        # 写一行CSV；列表各项分别成为一列；本行：writer.writerow(["path", "true_label", "predicted_label", "positive_probability"])
        writer.writerow(["path", "true_label", "predicted_label", "positive_probability"])
        # 把多条记录逐行写入CSV；本行：writer.writerows(predictions)
        writer.writerows(predictions)
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。metrics = {
    metrics = {
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。"model": "ResNet18 ImageNet features + new binary head",
        "model": "ResNet18 ImageNet features + new binary head",
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。"mode": "frozen backbone including BN statistics; train classification head only",
        "mode": "frozen backbone including BN statistics; train classification head only",
        # 随机种子，控制划分、初始化等随机过程；本行赋值/累计："seed": SEED, "max_epochs": epochs, "epochs": len(history), "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "seed": SEED, "max_epochs": epochs, "epochs": len(history), "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        # 数字标签0/1与类别名称的对应；本行赋值/累计："best_epoch_by_validation_loss": best_epoch, "class_names": CLASS_NAMES,
        "best_epoch_by_validation_loss": best_epoch, "class_names": CLASS_NAMES,
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。"train_count": 280, "validation_count": 40, "test_count": 80,
        "train_count": 280, "validation_count": 40, "test_count": 80,
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。"test_loss": test_loss, "test_accuracy": test_acc,
        "test_loss": test_loss, "test_accuracy": test_acc,
        # 行是真实类别、列是预测类别的2×2计数矩阵；本行赋值/累计："confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。"torch_version": str(torch.__version__), "python_executable": sys.executable,
        "torch_version": str(torch.__version__), "python_executable": sys.executable,
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。"limitation": "Image-level split without specimen IDs; educational run, not paper reproduction",
        "limitation": "Image-level split without specimen IDs; educational run, not paper reproduction",
    # 结束上方尚未闭合的参数、列表或字典；仍属于同一条语句，不是新增训练步骤。
    }
    # 把Python结构编码为JSON文本；indent是缩进，ensure_ascii=False保留中文；本行：(output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    # 生成四种图并保存；show_plots控制是否保留图窗；本行：save_plots(history, confusion, predictions, output_dir, show_plots=SHOW_PLOTS)
    save_plots(history, confusion, predictions, output_dir, show_plots=SHOW_PLOTS)
    # 定位本工程下的latest.json并写入最近一次完成的结果目录；以后--predict据此找到模型。
    (PROJECT / "runs" / "resnet18" / "latest.json").write_text(
        # 把Python结构编码为JSON文本；indent是缩进，ensure_ascii=False保留中文；本行：json.dumps({"run_dir": str(output_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")
        json.dumps({"run_dir": str(output_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"完成：最佳轮次={best_epoch}，测试准确率={test_acc:.1%}，测试损失={test_loss:.4f}", flush=True)
    print(f"完成：最佳轮次={best_epoch}，测试准确率={test_acc:.1%}，测试损失={test_loss:.4f}", flush=True)
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"模型、曲线、逐图预测均已保存到：{output_dir}", flush=True)
    print(f"模型、曲线、逐图预测均已保存到：{output_dir}", flush=True)
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print("请用验证集比较设置；保留测试集用于最终报告。", flush=True)
    print("请用验证集比较设置；保留测试集用于最终报告。", flush=True)
    # 判断条件；仅条件成立时执行下面缩进的分支。if SHOW_PLOTS:
    if SHOW_PLOTS:
        # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print("结果已保存。现在弹出四张图；全部关掉后终端会回到PS提示符。", flush=True)
        print("结果已保存。现在弹出四张图；全部关掉后终端会回到PS提示符。", flush=True)
        # 打开事件循环并等待关窗；此前结果已经保存；本行：plt.show()  # 保持图窗响应，直到用户关闭；保存动作已在此之前完成。
        plt.show()  # 保持图窗响应，直到用户关闭；保存动作已在此之前完成。
    # 当前面条件不成立时执行这个分支；具体目的看对应if。
    else:
        # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print("已按--no-show-plots仅保存图片。", flush=True)
        print("已按--no-show-plots仅保存图片。", flush=True)


# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def predict_one(image_path):
def predict_one(image_path):
    """对任意一张图片作答，读取最近一次完成实验的模型，不再训练。"""
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。latest = PROJECT / "runs" / "resnet18" / "latest.json"
    latest = PROJECT / "runs" / "resnet18" / "latest.json"
    # 判断条件；仅条件成立时执行下面缩进的分支。if not latest.is_file():
    if not latest.is_file():
        # 条件不满足时主动停止并报告原因，避免拿错误数据继续得到假结果。raise SystemExit("还没有完成的训练结果。请先不带 --predict 运行训练。")
        raise SystemExit("还没有完成的训练结果。请先不带 --predict 运行训练。")
    # 把JSON文本解析为Python字典；本行：output_dir = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"])
    output_dir = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"])
    # 读取自己保存的模型；map_location指定CPU，weights_only=True限制反序列化内容；本行：checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。model = build_model(use_pretrained=False)
    model = build_model(use_pretrained=False)
    # 把保存的参数装回当前模型；此后评估使用最佳轮次的参数；本行：model.load_state_dict(checkpoint["model_state"])
    model.load_state_dict(checkpoint["model_state"])
    # 保持特征提取器包括BatchNorm统计为评估状态；不等于禁止计算梯度；本行：model.eval()
    model.eval()
    # 进入受管理的代码块；结束时关闭文件/图片或恢复梯度开关，不需要手动善后。with Image.open(image_path) as image:
    with Image.open(image_path) as image:
        # 取得该预训练模型匹配的缩放、裁剪、张量转换和归一化；本行：tensor = WEIGHTS.transforms()(image.convert("RGB")).unsqueeze(0)  # 增加批次维度。
        tensor = WEIGHTS.transforms()(image.convert("RGB")).unsqueeze(0)  # 增加批次维度。
    # 此代码块只预测，不记录梯度，不更新参数；本行：with torch.no_grad():
    with torch.no_grad():
        # 把类别分数转成和为1的概率；dim=1为类别维；只展示，不再送入交叉熵。
        probabilities = model(tensor).softmax(dim=1)[0]
    # 把几个序列逐项配对，供同一条记录共同使用；本行：for name, probability in zip(checkpoint["class_names"], probabilities.tolist()):
    for name, probability in zip(checkpoint["class_names"], probabilities.tolist()):
        # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print(f"{name}：{probability:.1%}")
        print(f"{name}：{probability:.1%}")
    # 沿类别维度挑分数最高的编号；dim=1表示每张图在两个类别之间选择。
    print("预测类别：", checkpoint["class_names"][probabilities.argmax().item()])
    # 向终端输出进度或结果；flush=True表示及时显示，不等待缓冲。print("已知训练图上的预测只演示用法，不能代替独立测试。概率也不是可靠性保证。")
    print("已知训练图上的预测只演示用法，不能代替独立测试。概率也不是可靠性保证。")


# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def main():
def main():
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。parser = argparse.ArgumentParser(description="本机ResNet18预训练特征二分类教学")
    parser = argparse.ArgumentParser(description="本机ResNet18预训练特征二分类教学")
    # 注册命令行选项；type控制转换类型，default为默认值，help是--help文字。parser.add_argument("--epochs", type=int, default=None, help="覆盖config.json中的最大训练轮数")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖config.json中的最大训练轮数")
    # 注册命令行选项；type控制转换类型，default为默认值，help是--help文字。parser.add_argument("--no-show-plots", action="store_true", help="只保存图片，不弹窗")
    parser.add_argument("--no-show-plots", action="store_true", help="只保存图片，不弹窗")
    # 注册命令行选项；type控制转换类型，default为默认值，help是--help文字。parser.add_argument("--predict", type=Path, help="给出图片路径：只预测，不训练")
    parser.add_argument("--predict", type=Path, help="给出图片路径：只预测，不训练")
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。args = parser.parse_args()
    args = parser.parse_args()
    # 判断条件；仅条件成立时执行下面缩进的分支。if args.epochs is None:
    if args.epochs is None:
        # 把JSON文本解析为Python字典；本行：config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8-sig"))
        config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8-sig"))
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。args.epochs = int(config["max_epochs"])
        args.epochs = int(config["max_epochs"])
    # 判断条件；仅条件成立时执行下面缩进的分支。if args.epochs < 1:
    if args.epochs < 1:
        # 参数不合法时显示说明并退出，避免用错误轮数执行；本行：parser.error("--epochs 必须至少为1")
        parser.error("--epochs 必须至少为1")
    # 执行本文件上方的随机种子设置函数，不启动训练；本行：set_seed()
    set_seed()
    # 判断条件；仅条件成立时执行下面缩进的分支。if args.predict:
    if args.predict:
        # 进入单张图片预测分支；不更新参数、不训练新模型；本行：predict_one(args.predict)
        predict_one(args.predict)
    # 当前面条件不成立时执行这个分支；具体目的看对应if。
    else:
        # 进入完整训练流程；仅在用户实际运行入口且没有--predict时调用；本行：train(args.epochs)
        train(args.epochs)


# Windows下尤其建议保留这个入口保护，避免导入模块时意外开始训练。
# 判断条件；仅条件成立时执行下面缩进的分支。if __name__ == "__main__":
if __name__ == "__main__":
    # 调用本脚本的命令行入口，先解析选项再选择训练或预测；本行：main()
    main()
