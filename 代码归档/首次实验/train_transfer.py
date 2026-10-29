r"""第二课：使用 ImageNet 预训练 ResNet18 的特征，训练碳纤维二分类头。

操作：在工程根目录创建 train_transfer.py，把本参考文件的全部内容复制进去。
运行：.\.venv\Scripts\python.exe .\train_transfer.py
首次运行会从 PyTorch 官方地址下载预训练权重，之后计算均在本机进行。

本课只训练新的全连接层，不更新卷积特征提取器，也不是原论文的完整复现。
"""

# ==================== 1. 导入工具 ====================
import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    
    matplotlib.use("Agg")  # 把图保存成文件，避免弹出窗口打断训练。
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:
    raise SystemExit(
        "依赖尚未装好。请按 README 的顺序安装 PyTorch 和 requirements.txt。\n"
        f"当前解释器：{sys.executable}\n原始错误：{exc}"
    ) from exc

# 既能从工程根目录运行，也能找到参考代码所在工程。
PROJECT = next(
    (p for p in Path(__file__).resolve().parents if (p / "config.json").is_file()),
    None,
)
if PROJECT is None:
    raise SystemExit("找不到 config.json，请将代码保存在缺陷检测学习工程内。")

SEED = 20261029
EPOCHS = 5                 # 默认完整看训练集5遍，可在运行时用 --epochs 修改。
BATCH_SIZE = 16            # 一次处理16张图片，最后一个批次可以少于16张。
LEARNING_RATE = 0.001      # 更新分类头参数的步长。
CLASS_NAMES = ["negative", "positive"]  # 0=负类，1=正类；不要理解为消极/积极。
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1

# 让预训练权重缓存也保存在D盘工程，而不是默认的C盘用户目录。
torch.hub.set_dir(str(PROJECT / "weights"))


def set_seed():
    """固定随机种子，方便重跑比较；不保证不同硬件/版本逐位完全一致。"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)  # 第一课使用CPU，避免同时开过多线程。


def prepare_splits(data_dir):
    """读取标签并划分：每类140训练、20验证、40测试，总计280/40/80。

    先划分，再做各自预处理。第一轮不增加随机增强，方便理解和复现。
    这里检查字节完全相同的图，避免它们混入不同集合；近似图和同工件
    不同视角仍可能存在，所以这只是图片级教学评估，不能证明跨工件泛化。
    """
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(SEED)
    seen_hashes = {}
    for label, class_name in enumerate(CLASS_NAMES):
        folder = data_dir / class_name
        files = sorted(
            p for p in folder.glob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(files) != 200:
            raise ValueError(f"{folder} 应有200张图片，实际找到{len(files)}张。")
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError(f"发现完全重复图片，请先核查：{path} 与 {seen_hashes[digest]}")
            seen_hashes[digest] = path
        rng.shuffle(files)
        splits["train"].extend((p, label) for p in files[:140])
        splits["validation"].extend((p, label) for p in files[140:160])
        splits["test"].extend((p, label) for p in files[160:])
    return splits


class CarbonDataset(Dataset):
    """告诉PyTorch：怎样根据一个编号，读取图片及其标准答案。"""
    def __init__(self, records):
        self.records = records
        # 使用这套预训练权重配套的缩放、中心裁剪和归一化。
        # 输出从PIL图片变成张量，单张形状为[3, 224, 224]。
        # 归一化常数来自预训练配置，不从测试集重新估计。
        self.transform = WEIGHTS.transforms()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, label, str(path)


def build_model(use_pretrained):
    """保留预训练特征，替换分类层。预测已有模型时无需再次下载权重。"""
    model = resnet18(weights=WEIGHTS if use_pretrained else None)
    # 冻结旧参数：后续反向传播不计算它们的梯度，也不更新这些权重。
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 原分类层对应ImageNet的1000类；我们改为两个类别。
    # ResNet18汇总后有512个特征，新层有512*2+2=1026个可训练参数。
    # 新建Linear层的参数默认可以训练。
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def run_one_phase(model, loader, criterion, device, optimizer=None):
    """有optimizer时训练；没有时只评估。返回损失、准确率、逐图输出。"""
    training = optimizer is not None
    # 特征提取器被冻结，BatchNorm的运行统计也保持不变。
    # requires_grad=False只冻结权重，单独调用model.train()还会改变BN统计，
    # 所以本课保持整个模型eval，只将新分类层设为训练状态。
    model.eval()
    model.fc.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    outputs = []

    # train用梯度；validation/test不记录梯度，也不更新参数。
    with torch.set_grad_enabled(training):
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()       # 清空上一批次留下的梯度。
            logits = model(images)          # 前向传播：得到两个原始类别分数。
            # CrossEntropyLoss内部已处理对应的log-softmax。
            # 因此这里直接传logits，不能先softmax再交给这个损失函数。
            loss = criterion(logits, labels)
            if training:
                loss.backward()             # 反向传播：算分类头参数的梯度。
                optimizer.step()            # 用梯度更新参数；旧卷积参数不变。

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            count += batch_size
            # softmax只用于展示概率形式的输出，不用于上面的损失输入。
            probabilities = logits.detach().softmax(dim=1).cpu().tolist()
            for path, truth, guess, probs in zip(
                paths, labels.cpu().tolist(), predictions.cpu().tolist(), probabilities
            ):
                outputs.append((path, truth, guess, probs[1]))
    return loss_sum / count, correct / count, outputs


def save_plots(history, confusion, predictions, output_dir):
    """保存四类结果图。混淆矩阵的行是真实类别，列是预测类别。"""
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for phase in ["train", "validation"]:
        axes[0].plot(epochs, [r[f"{phase}_loss"] for r in history], marker="o", label=phase)
        axes[1].plot(epochs, [r[f"{phase}_accuracy"] for r in history], marker="o", label=phase)
    for ax, title in zip(axes, ["Cross-entropy loss", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(confusion[row][col]), ha="center", va="center",
                    color="white" if confusion[row][col] > np.max(confusion) / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


    # 图3：每类指标。precision看“判成该类的有多少是真的”；
    # recall看“真实该类有多少被找回”；F1综合两者。
    per_class = []
    for label in range(2):
        tp = int(confusion[label, label])
        predicted_count = int(confusion[:, label].sum())
        true_count = int(confusion[label, :].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / true_count if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(2)
    for offset, (index, name) in zip([-0.24, 0, 0.24], enumerate(["Precision", "Recall", "F1"])):
        bars = ax.bar(positions + offset, [values[index] for values in per_class], width=0.24, label=name)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(positions, CLASS_NAMES)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-class test metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=160)
    plt.close(fig)

    # 图4：直接看测试图。优先展示最多6张错误，再补正确图，总计最多9张。
    # 这是按错误优先选择的示例，不是测试集错误率的可视化估计。
    wrong = [r for r in predictions if r[1] != r[2]]
    right = [r for r in predictions if r[1] == r[2]]
    examples = wrong[:6] + right[:9 - min(6, len(wrong))]
    if len(examples) < 9:
        examples += wrong[6:6 + 9 - len(examples)]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (path, truth, guess, probability) in zip(axes.flat, examples):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"True: {CLASS_NAMES[truth]} | Pred: {CLASS_NAMES[guess]}\n"
            f"P(positive)={probability:.2f} | {Path(path).name}",
            color="darkgreen" if truth == guess else "firebrick", fontsize=9,
        )
    fig.suptitle("Test examples: errors first (selected examples, not a random sample)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "test_examples.png", dpi=160)
    plt.close(fig)

def train(epochs):
    """一次完整流程：读数据→划分→加载模型→训练/验证→最终测试→保存。"""
    config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8-sig"))
    data_dir = Path(config["binary_data"])
    splits = prepare_splits(data_dir)
    # 每次新建结果目录，不覆盖以前的实验。
    output_dir = PROJECT / "runs" / "resnet18" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "split.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_name", "split"])
        for phase, records in splits.items():
            writer.writerows((str(p), label, CLASS_NAMES[label], phase) for p, label in records)

    loaders = {
        phase: DataLoader(CarbonDataset(records), batch_size=BATCH_SIZE,
                          shuffle=(phase == "train"), num_workers=0)
        for phase, records in splits.items()
    }
    device = torch.device("cpu")  # 第一遍先用CPU跑通，不依赖显卡和CUDA配置。
    print(f"Python：{sys.executable}", flush=True)
    print(f"数据：{data_dir}\n划分：280训练 / 40验证 / 80测试", flush=True)
    print(f"计算设备：{device}，结果目录：{output_dir}", flush=True)
    print("正在加载ImageNet预训练权重，第一次运行需要联网下载……", flush=True)
    model = build_model(use_pretrained=True).to(device)
    print("仅训练新的全连接层，参数数目：",
          sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    history, best_loss, best_epoch = [], float("inf"), 0

    # 训练期间不评估测试集；只根据验证损失选择最佳轮次。
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_one_phase(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc, _ = run_one_phase(model, loaders["validation"], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        print(f"Epoch {epoch}/{epochs} | train loss={train_loss:.4f}, acc={train_acc:.1%}"
              f" | validation loss={val_loss:.4f}, acc={val_acc:.1%}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save({
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "class_names": CLASS_NAMES, "weights_name": "IMAGENET1K_V1", "epoch": epoch,
            }, output_dir / "best_model.pt")
        with (output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    # 训练结束，重新装入验证集选中的参数，再对80张测试图评估一次。
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, predictions = run_one_phase(model, loaders["test"], criterion, device)
    confusion = np.zeros((2, 2), dtype=int)
    for _, truth, guess, _ in predictions:
        confusion[truth, guess] += 1
    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "predicted_label", "positive_probability"])
        writer.writerows(predictions)
    metrics = {
        "model": "ResNet18 ImageNet features + new binary head",
        "mode": "frozen backbone including BN statistics; train classification head only",
        "seed": SEED, "epochs": epochs, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "best_epoch_by_validation_loss": best_epoch, "class_names": CLASS_NAMES,
        "train_count": 280, "validation_count": 40, "test_count": 80,
        "test_loss": test_loss, "test_accuracy": test_acc,
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "torch_version": str(torch.__version__), "python_executable": sys.executable,
        "limitation": "Image-level split without specimen IDs; educational run, not paper reproduction",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(history, confusion, predictions, output_dir)
    (PROJECT / "runs" / "resnet18" / "latest.json").write_text(
        json.dumps({"run_dir": str(output_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：最佳轮次={best_epoch}，测试准确率={test_acc:.1%}，测试损失={test_loss:.4f}", flush=True)
    print(f"模型、曲线、逐图预测均已保存到：{output_dir}", flush=True)
    print("这只是一次教学实验；不要根据测试结果反复调参。", flush=True)


def predict_one(image_path):
    """对任意一张图片作答，读取最近一次完成实验的模型，不再训练。"""
    latest = PROJECT / "runs" / "resnet18" / "latest.json"
    if not latest.is_file():
        raise SystemExit("还没有完成的训练结果。请先不带 --predict 运行训练。")
    output_dir = Path(json.loads(latest.read_text(encoding="utf-8"))["run_dir"])
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = build_model(use_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with Image.open(image_path) as image:
        tensor = WEIGHTS.transforms()(image.convert("RGB")).unsqueeze(0)  # 增加批次维度。
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0]
    for name, probability in zip(checkpoint["class_names"], probabilities.tolist()):
        print(f"{name}：{probability:.1%}")
    print("预测类别：", checkpoint["class_names"][probabilities.argmax().item()])
    print("已知训练图上的预测只演示用法，不能代替独立测试。概率也不是可靠性保证。")


def main():
    parser = argparse.ArgumentParser(description="本机ResNet18预训练特征二分类教学")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数，默认5")
    parser.add_argument("--predict", type=Path, help="给出图片路径：只预测，不训练")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须至少为1")
    set_seed()
    if args.predict:
        predict_one(args.predict)
    else:
        train(args.epochs)


# Windows下尤其建议保留这个入口保护，避免导入模块时意外开始训练。
if __name__ == "__main__":
    main()



