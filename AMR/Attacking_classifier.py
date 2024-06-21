from rfml.attack import fgsm
from rfml.data import build_dataset
from rfml.nn.eval import compute_accuracy, compute_accuracy_on_cross_sections, compute_confusion
from rfml.nn.model import build_model
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from rfml_project.rfml.attack import pgd

_, _, test, le = build_dataset(dataset_name="RML2016.10a", path="D:\RML2016.10a_dict.pkl", test_pct=0.9)#set path to your project directory
mask = test.df["SNR"] >= 18
model = build_model(model_name="cnn", input_samples=128, n_classes=len(le))
model.load("CNN.pt")


acc = compute_accuracy(model=model, data=test, le=le, mask=mask)
print("没有攻击的准确率: {:.3f}".format(acc))

spr = 10  # dB
right = 0
total = 0
preds = []
actual_labels = []
dl = DataLoader(test.as_torch(le=le, mask=mask), shuffle=True, batch_size=512)
for x, y in dl:
    # adv_x = fgsm(x, y, spr=spr, input_size=128, sps=8, net=model)
    adv_x = pgd(x, y, k=15, input_size=128, net=model)
    predictions = model.predict(adv_x)
    preds.extend(predictions.tolist())
    actual_labels.extend(y.tolist())
    right += (predictions == y).sum().item()
    total += len(y)
adv_acc = float(right) / total
# 将列表转换为NumPy数组以便后续处理
predictions = np.array(preds)
actual_labels = np.array(actual_labels)

print("加入噪声后准确率: {:.3f}".format( adv_acc))
print("准确率下降了 {:.3f}".format(acc - adv_acc))

# 使用sklearn的confusion_matrix函数生成混淆矩阵
cm = confusion_matrix(actual_labels, predictions, labels=range(len(le)))
# 计算每行的总和
row_sums = cm.sum(axis=1, keepdims=True)
# 避免除以零的情况（虽然在实际混淆矩阵中不太可能出现全零行）
row_sums[row_sums == 0] = 1
# 计算每个元素占其所在行的百分比
percentage_matrix = (cm / row_sums) * 1.0

# 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(percentage_matrix, annot=True, fmt='.2%', cmap='PuRd', xticklabels=le.labels, yticklabels=le.labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CNN+LSTM&PGD')
# 保存图像为 PNG 格式
# plt.savefig('CNN+LSTM&PGD.png')
plt.show()
