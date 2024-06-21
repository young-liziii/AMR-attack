import torch
import torch.nn as nn

from .base import Model
# from rfml.nn.layers import Flatten, PowerNormalization
from ..layers import Flatten, PowerNormalization


class ALTERNATIVE_MODEL(Model):


    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)
        self.preprocess = PowerNormalization()

        # Batch x 1-channel x IQ x input_samples(128)  Batch_size默认512
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=False,
        )
        self.a1 = nn.ReLU()
        # 批量归一化层，针对50个输出通道。
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)

        # 展平层，但保留时间维度，这意味着输入的批次和时间维度不会被合并
        self.flatten_preserve_time = Flatten(preserve_time=True)

        self.lstm_layers = 2
        self.lstm_directions = True
        self.lstm_hidden_size = n_classes
        self.lstm = nn.LSTM(input_size=128 * 2,  # * IQ (2)
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            batch_first=True,
                            bidirectional=self.lstm_directions)

        self.flatten = Flatten()

        self.dense1 = nn.Linear(
            input_samples * self.lstm_hidden_size * 2, 256
        )
        self.a4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(256)

        self.dense2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.preprocess(x)

        x = self.conv1(x)
        x = self.a1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.a2(x)
        x = self.bn2(x)


        x = self.flatten_preserve_time(x)  # BxTxF
        x, _ = self.lstm(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.a4(x)
        x = self.bn4(x)

        x = self.dense2(x)

        return x

