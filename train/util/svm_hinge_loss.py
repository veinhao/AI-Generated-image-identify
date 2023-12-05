# TODO 奇怪的问题，为什么会传入b
def hinge_loss(b, outputs, labels):
    """
    合页损失函数 for svm head of model
    参数:
    outputs - 模型的预测输出
    labels - 真实标签
    """

    # 选择正类的得分
    outputs = outputs[:, 1]  # 假设正类是第二列

    labels = 2 * labels - 1  # 将标签从 [0, 1] 转换为 [-1, 1]
    loss = 1 - outputs * labels
    loss[loss < 0] = 0  # max(0, 1 - y * f(x))
    return loss.mean()
