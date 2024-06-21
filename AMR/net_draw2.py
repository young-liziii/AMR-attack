from prettytable import PrettyTable
import torch
from rfml.nn.model import build_model

# 假设我们有128个采样点和11个类别
input_samples = 128
n_classes = 11

# 构建模型
model = build_model(model_name="alternative_model", input_samples=input_samples, n_classes=n_classes)

# 创建prettytable对象
pt = PrettyTable()
pt.field_names = ["Layer (type)", "Input Shape", "Output Shape", "Param #", "Parameter Shapes"]

# 模拟输入数据，形状为 [512, 1, 2, 128]
input_data = torch.randn(512, 1, 2, 128)

print("Model Architecture:")
prev_layer_output_shape = (512, 1, 2, 128)  # 假设的输入形状

for name, layer in model.named_children():
    # 前向传播以获取当前层的输出形状
    output_data = layer(input_data)

    # 检查输出是否为元组
    if isinstance(output_data, tuple):
        # 如果是元组，取第一个元素作为输出数据
        output_data = output_data[0]

    # 记录层的输入形状和输出形状
    input_shape = prev_layer_output_shape
    output_shape = output_data.shape

    # 添加层信息到表格
    layer_param_shapes = [p.shape for p in layer.parameters()]
    pt.add_row([name, input_shape, output_shape, sum(p.numel() for p in layer.parameters()), layer_param_shapes])

    # 更新前一层的输出形状为当前层的输出形状，以便下一层次的计算
    prev_layer_output_shape = output_shape

    input_data = output_data  # 将当前层的输出作为下一层的输入

# 打印表格
print(pt)

# 计算总参数数量
total_params = sum(torch.numel(param) for name, param in model.named_parameters())
print("\nTotal Parameters in Model: {}".format(total_params))