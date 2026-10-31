"""数据模块：查找图片、生成划分，以及把一张图片变成模型输入。

prepare_splits(data_dir, seed)返回train/validation/test三组记录。
CarbonDataset根据编号返回图片张量、标签、路径。
只读取原图，不移动原图；本模块不安装库或运行训练。
"""
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import hashlib  # 用文件内容摘要检查完全重复的图片。
import hashlib  # 用文件内容摘要检查完全重复的图片。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import random  # 使用独立随机数生成器复现相同的数据划分。
import random  # 使用独立随机数生成器复现相同的数据划分。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from PIL import Image  # Pillow负责读取图片。
from PIL import Image  # Pillow负责读取图片。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from torch.utils.data import Dataset  # 自定义数据集需要继承的接口。
from torch.utils.data import Dataset  # 自定义数据集需要继承的接口。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from model import CLASS_NAMES, WEIGHTS  # 标签顺序和预处理与模型使用同一套定义。
from model import CLASS_NAMES, WEIGHTS  # 标签顺序和预处理与模型使用同一套定义。

# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def prepare_splits(data_dir, seed):
def prepare_splits(data_dir, seed):
    """读取标签并划分：每类140训练、20验证、40测试，总计280/40/80。

    先划分，再做各自预处理。第一轮不增加随机增强，方便理解和复现。
    这里检查字节完全相同的图，避免它们混入不同集合；近似图和同工件
    不同视角仍可能存在，所以这只是图片级教学评估，不能证明跨工件泛化。
    """
    # 训练/验证/测试三组路径和标签；本行赋值/累计：splits = {"train": [], "validation": [], "test": []}
    splits = {"train": [], "validation": [], "test": []}
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。rng = random.Random(seed)
    rng = random.Random(seed)
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。seen_hashes = {}
    seen_hashes = {}
    # 逐个遍历并同时得到从0开始的编号；本行：for label, class_name in enumerate(CLASS_NAMES):
    for label, class_name in enumerate(CLASS_NAMES):
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。folder = data_dir / class_name
        folder = data_dir / class_name
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。files = sorted(
        files = sorted(
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。p for p in folder.glob("*")
            p for p in folder.glob("*")
            # 判断条件；仅条件成立时执行下面缩进的分支。if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        # 结束上方尚未闭合的参数、列表或字典；仍属于同一条语句，不是新增训练步骤。
        )
        # 判断条件；仅条件成立时执行下面缩进的分支。if len(files) != 200:
        if len(files) != 200:
            # 条件不满足时主动停止并报告原因，避免拿错误数据继续得到假结果。raise ValueError(f"{folder} 应有200张图片，实际找到{len(files)}张。")
            raise ValueError(f"{folder} 应有200张图片，实际找到{len(files)}张。")
        # 开始循环：逐项取出数据；下方缩进内容重复执行。for path in files:
        for path in files:
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # 判断条件；仅条件成立时执行下面缩进的分支。if digest in seen_hashes:
            if digest in seen_hashes:
                # 条件不满足时主动停止并报告原因，避免拿错误数据继续得到假结果。raise ValueError(f"发现完全重复图片，请先核查：{path} 与 {seen_hashes[digest]}")
                raise ValueError(f"发现完全重复图片，请先核查：{path} 与 {seen_hashes[digest]}")
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。seen_hashes[digest] = path
            seen_hashes[digest] = path
        # 使用该随机数生成器原地打乱路径顺序；不复制或修改图片；本行：rng.shuffle(files)
        rng.shuffle(files)
        # 将路径与数字标签组成二元组加入对应分组；每类前140张用于训练，切片右端不包含，不复制图片。
        splits["train"].extend((p, label) for p in files[:140])
        # 将路径与数字标签组成二元组加入对应分组；每类第140至159张用于验证，切片右端不包含，不复制图片。
        splits["validation"].extend((p, label) for p in files[140:160])
        # 将路径与数字标签组成二元组加入对应分组；每类第160张起的剩余图片用于测试，切片右端不包含，不复制图片。
        splits["test"].extend((p, label) for p in files[160:])
    # 把这些值按顺序交还调用者；接收变量的顺序必须一致。return splits
    return splits


# 定义自定义数据集类；继承Dataset以便DataLoader按编号读取样本。
class CarbonDataset(Dataset):
    """告诉PyTorch：怎样根据一个编号，读取图片及其标准答案。"""
    # 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def __init__(self, records):
    def __init__(self, records):
        # 本数据集的(路径,标签)列表；本行赋值/累计：self.records = records
        self.records = records
        # 使用这套预训练权重配套的缩放、中心裁剪和归一化。
        # 输出从PIL图片变成张量，单张形状为[3, 224, 224]。
        # 归一化常数来自预训练配置，不从测试集重新估计。
        # 取得该预训练模型匹配的缩放、裁剪、张量转换和归一化；本行：self.transform = WEIGHTS.transforms()
        self.transform = WEIGHTS.transforms()

    # 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def __len__(self):
    def __len__(self):
        # 把这些值按顺序交还调用者；接收变量的顺序必须一致。return len(self.records)
        return len(self.records)

    # 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def __getitem__(self, index):
    def __getitem__(self, index):
        # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。path, label = self.records[index]
        path, label = self.records[index]
        # 进入受管理的代码块；结束时关闭文件/图片或恢复梯度开关，不需要手动善后。with Image.open(path) as image:
        with Image.open(path) as image:
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。image_tensor = self.transform(image.convert("RGB"))
            image_tensor = self.transform(image.convert("RGB"))
        # 把这些值按顺序交还调用者；接收变量的顺序必须一致。return image_tensor, label, str(path)
        return image_tensor, label, str(path)
