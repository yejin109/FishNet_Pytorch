import torch.nn as nn


class FishBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, mode='DR', k=1, dilation=1):
        super(FishBlock, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU()
        self.k = k

        bottle_neck_ch = ch_out // 4

        self.bn1 = nn.BatchNorm2d(ch_in)
        self.conv1 = nn.Conv2d(ch_in, bottle_neck_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bottle_neck_ch)
        self.conv2 = nn.Conv2d(bottle_neck_ch, bottle_neck_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,bias=False)

        self.bn3 = nn.BatchNorm2d(bottle_neck_ch)
        self.conv3 = nn.Conv2d(bottle_neck_ch, ch_out, kernel_size=1, bias=False)

        if mode == 'UR':
            self.shortcut = None
        elif ch_in != ch_out or stride > 1:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(ch_in),
                self.relu,
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = None

    def channel_wise_reduction(self, data):

        n, c, h, w = data.size()
        return data.view(n, c//self.k, self.k, h, w).sum(2)

    def forward(self, data):

        out = self.conv1(self.relu(self.bn1(data)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.mode == 'UR':
            residual = self.channel_wise_reduction(data)
        elif self.shortcut is not None:
            residual = self.shortcut(data)
        else:
            residual = data

        out += residual
        return out
