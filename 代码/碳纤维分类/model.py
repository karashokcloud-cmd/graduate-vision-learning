"""模型模块：负责加载特征提取器，以及创建新的二分类层。

输入：是否加载ImageNet预训练权重。
输出：ResNet18模型；旧参数冻结，新分类层可训练。
本模块不启动训练；第一次真正调用预训练模型时才可能下载权重。
"""
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from torch import nn  # nn提供Linear等网络层。
from torch import nn  # nn提供Linear等网络层。
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。from torchvision.models import ResNet18_Weights, resnet18  # 官方权重说明与网络构造函数。
from torchvision.models import ResNet18_Weights, resnet18  # 官方权重说明与网络构造函数。

# 数字标签0/1与类别名称的对应；本行赋值/累计：CLASS_NAMES = ["negative", "positive"]  # 列表下标就是数字标签0、1。
CLASS_NAMES = ["negative", "positive"]  # 列表下标就是数字标签0、1。
# 固定的ImageNet预训练权重版本及其配套预处理；本行赋值/累计：WEIGHTS = ResNet18_Weights.IMAGENET1K_V1  # 固定权重版本，避免随默认值变化。
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1  # 固定权重版本，避免随默认值变化。

# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def build_model(use_pretrained):
def build_model(use_pretrained):
    """保留预训练特征，替换分类层。预测已有模型时无需再次下载权重。"""
    # 构造ResNet18；weights为固定预训练版本或None，None不下载；本行：model = resnet18(weights=WEIGHTS if use_pretrained else None)
    model = resnet18(weights=WEIGHTS if use_pretrained else None)
    # 冻结旧参数：后续反向传播不计算它们的梯度，也不更新这些权重。
    # 开始循环：逐项取出数据；下方缩进内容重复执行。for parameter in model.parameters():
    for parameter in model.parameters():
        # 冻结这个原有参数，后续不为它计算训练梯度，也不会由分类层优化器更新。
        parameter.requires_grad = False

    # 原分类层对应ImageNet的1000类；我们改为两个类别。
    # ResNet18汇总后有512个特征，新层有512*2+2=1026个可训练参数。
    # 新建Linear层的参数默认可以训练。
    # Linear(in_features,out_features)：把512个特征加权成2个分数；默认含2个偏置；本行：model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    # 把这些值按顺序交还调用者；接收变量的顺序必须一致。return model
    return model
