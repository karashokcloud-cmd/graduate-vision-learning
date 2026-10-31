"""训练与评估模块：执行一个完整阶段，不决定读哪些数据或保存到哪里。

有optimizer时：前向计算、损失、反向传播、参数更新。
没有optimizer时：只评估，不更新参数。
返回平均损失、准确率、逐张图片的预测记录。
"""
# 导入后面需要的模块或名称；import只加载已经安装/已有的代码，不会自动安装库。import torch  # 控制是否记录梯度，以及执行张量计算。
import torch  # 控制是否记录梯度，以及执行张量计算。

# 定义函数，不在这里执行；括号内是调用者传入的参数，=右边是省略该参数时的默认值。def run_one_phase(model, loader, criterion, device, optimizer=None):
def run_one_phase(model, loader, criterion, device, optimizer=None):
    """有optimizer时训练；没有时只评估。返回损失、准确率、逐图输出。"""
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。training = optimizer is not None
    training = optimizer is not None
    # 特征提取器被冻结，BatchNorm的运行统计也保持不变。
    # requires_grad=False只冻结权重，单独调用model.train()还会改变BN统计，
    # 所以本课保持整个模型eval，只将新分类层设为训练状态。
    # 保持特征提取器包括BatchNorm统计为评估状态；不等于禁止计算梯度；本行：model.eval()
    model.eval()
    # 只切换新分类层训练状态；training为True或False；本行：model.fc.train(training)
    model.fc.train(training)
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。loss_sum, correct, count = 0.0, 0, 0
    loss_sum, correct, count = 0.0, 0, 0
    # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。outputs = []
    outputs = []

    # train用梯度；validation/test不记录梯度，也不更新参数。
    # 训练时记录梯度，验证测试时关闭梯度记录，减少开销；本行：with torch.set_grad_enabled(training):
    with torch.set_grad_enabled(training):
        # 开始循环：逐项取出数据；下方缩进内容重复执行。for images, labels, paths in loader:
        for images, labels, paths in loader:
            # 将图片、标签或模型放到指定计算设备；当前device是CPU，双方设备需要一致。
            images, labels = images.to(device), labels.to(device)
            # 判断条件；仅条件成立时执行下面缩进的分支。if training:
            if training:
                # 清空上一批梯度，防止意外累积；本行：optimizer.zero_grad()       # 清空上一批次留下的梯度。
                optimizer.zero_grad()       # 清空上一批次留下的梯度。
            # 形状为批大小×2的原始分数，还不是概率；本行赋值/累计：logits = model(images)          # 前向传播：得到两个原始类别分数。
            logits = model(images)          # 前向传播：得到两个原始类别分数。
            # CrossEntropyLoss内部已处理对应的log-softmax。
            # 因此这里直接传logits，不能先softmax再交给这个损失函数。
            # 本批样本的平均交叉熵损失；本行赋值/累计：loss = criterion(logits, labels)
            loss = criterion(logits, labels)
            # 判断条件；仅条件成立时执行下面缩进的分支。if training:
            if training:
                # 根据当前损失计算可训练参数梯度；此处不直接更新权重；本行：loss.backward()             # 反向传播：算分类头参数的梯度。
                loss.backward()             # 反向传播：算分类头参数的梯度。
                # 优化器读取梯度更新分类层参数；本行：optimizer.step()            # 用梯度更新参数；旧卷积参数不变。
                optimizer.step()            # 用梯度更新参数；旧卷积参数不变。

            # 沿类别维度挑分数最高的编号；dim=1表示每张图在两个类别之间选择。
            predictions = logits.argmax(dim=1)
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。batch_size = labels.size(0)
            batch_size = labels.size(0)
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。loss_sum += loss.item() * batch_size
            loss_sum += loss.item() * batch_size
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。correct += (predictions == labels).sum().item()
            correct += (predictions == labels).sum().item()
            # 把右边表达式的结果保存到左边名称；若左边有多个名称，则按顺序拆开返回值。count += batch_size
            count += batch_size
            # softmax只用于展示概率形式的输出，不用于上面的损失输入。
            # 把类别分数转成和为1的概率；dim=1为类别维；只展示，不再送入交叉熵。
            probabilities = logits.detach().softmax(dim=1).cpu().tolist()
            # 把几个序列逐项配对，供同一条记录共同使用；本行：for path, truth, guess, probs in zip(
            for path, truth, guess, probs in zip(
                # 将每张图的路径、真实标签、预测编号与概率逐项对齐；cpu后tolist变成普通Python列表。
                paths, labels.cpu().tolist(), predictions.cpu().tolist(), probabilities
            # 结束多行循环或函数调用头；下一段缩进属于该循环或调用关联的代码块。
            ):
                # 把一张图的路径、真实标签、预测标签和positive概率加入记录列表；本行：outputs.append((path, truth, guess, probs[1]))
                outputs.append((path, truth, guess, probs[1]))
    # 把这些值按顺序交还调用者；接收变量的顺序必须一致。return loss_sum / count, correct / count, outputs
    return loss_sum / count, correct / count, outputs
