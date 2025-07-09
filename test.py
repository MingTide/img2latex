import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 3 维的可训练向量
        self.my_param = nn.Parameter( torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x):
        # 模拟参数参与计算
        return x + self.my_param


model = MyModel()
x = torch.tensor([1.0, 2.0, 3.0])
output = model(x)
print("输出:", output)