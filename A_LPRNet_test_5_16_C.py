import torch.nn as nn
import torch
from torch.nn import functional as F

import torch.nn as nn
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = x
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * out


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.BatchNorm2d(num_features=ch_out // 4),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=ch_out // 4),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(num_features=ch_out // 4),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1)

        )
        self.conv3 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        out = self.block(x) + self.conv3(x)
        return out


class Residual(nn.Module):  # @save
    def __init__(self, class_num):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(class_num)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=class_num,
                               kernel_size=(1, 1))

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        X = self.conv3(X)
        Y += X
        return F.relu(Y)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2##

            ChannelAttention(in_planes=64),
            SpatialAttention(),

            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),

            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6##

            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),

            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10

            small_basic_block(ch_in=256, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10

            small_basic_block(ch_in=256, ch_out=512),  # *** 11 ***
            nn.BatchNorm2d(num_features=512),  # 12
            nn.ReLU(),  # 13##

            ChannelAttention(in_planes=512),
            SpatialAttention(),

            small_basic_block(ch_in=512, ch_out=512),  # *** 11 ***
            nn.BatchNorm2d(num_features=512),  # 12
            nn.ReLU(),  # 13##

            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14

            # nn.Dropout(dropout_rate),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18

            # nn.Dropout(dropout_rate),
            ChannelAttention(in_planes=256),
            SpatialAttention(),

            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(1, 1), stride=(1,1)),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU()  # *** 22 ***##

            # Residual(class_num=class_num)
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )
        self.ca = ChannelAttention(in_planes=448+self.class_num)
        self.sa = SpatialAttention()

    def forward(self, x):
        # keep_features = list()
        # for i, layer in enumerate(self.backbone.children()):
        #     x = layer(x)
        #     if i in [2, 6, 13, 20]:  # [2, 4, 8, 11, 22]
        #         keep_features.append(x)
        #
        # global_context = list()
        # for i, f in enumerate(keep_features):
        #     if i in [0, 1]:
        #         f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
        #     if i in [2]:
        #         f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
        #     f_pow = torch.pow(f, 2)
        #     f_mean = torch.mean(f_pow)
        #     f = torch.div(f, f_mean)
        #     global_context.append(f)
        #
        #
        # x = torch.cat(global_context, 1)
        x = self.backbone(x)
        # x = self.ca(x)
        # x = self.sa(x)
        # x = self.container(x)

        logits = torch.mean(x, dim=2)

        return logits


def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
