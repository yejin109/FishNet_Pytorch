import math
import torch
import torch.nn as nn
import sys
import os

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}')

from fish_block import FishBlock


class FishNet(nn.Module):
    def __init__(self, **kwargs):
        super(FishNet, self).__init__()

        ch_initial = kwargs['tail_ch_in'][0]
        self.layer1 = self.layers(3, ch_initial // 2, stride=2)
        self.layer2 = self.layers(ch_initial // 2, ch_initial // 2)
        self.layer3 = self.layers(ch_initial // 2, ch_initial)

        self.pool = nn.MaxPool2d(3, padding=1, stride=2)
        self.fish = Fish(**kwargs)
        self.init_weights()

    @staticmethod
    def layers(ch_in, ch_out, stride=1):
        result = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        return result

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data):
        out = self.layer1(data)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = self.fish(out)

        result = out.view(out.size(0), -1)
        return result


class Fish(nn.Module):
    def __init__(self, tail_ch_in, tail_ch_out, tail_res_blks, body_ch_in, body_ch_out, body_res_blks, body_trans_blks,
                 head_ch_in, head_ch_out, head_res_blks, head_trans_blks, num_cls):
        super(Fish, self).__init__()
        self.num_cls = num_cls

        self.num_tail = len(body_ch_out)
        self.tail_ch_in= tail_ch_in
        self.tail_ch_out = tail_ch_out
        self.tail_res_blks = tail_res_blks

        self.num_body = len(body_ch_out)
        self.body_ch_in = body_ch_in
        self.body_ch_out = body_ch_out
        self.body_res_blks = body_res_blks
        self.body_trans_blks = body_trans_blks

        self.num_head = len(head_ch_out)
        self.head_ch_in = head_ch_in
        self.head_ch_out = head_ch_out
        self.head_res_blks = head_res_blks
        self.head_trans_blks = head_trans_blks
        self.tail, self.se, self.body, self.head, self.score = self.make_fish()

    def make_fish(self):
        tail = make_blocks(self.num_tail, self.tail_ch_in, self.tail_ch_out, self.tail_res_blks, 'tail')

        se = make_se_block(self.tail_ch_out[-1], self.tail_ch_out[-1])

        body = make_blocks(self.num_body, self.body_ch_in, self.body_ch_out, self.body_res_blks, 'body')

        head = make_blocks(self.num_head, self.head_ch_in, self.head_ch_out, self.head_res_blks, 'head')

        score = make_score(self.head_ch_out[-1]+self.tail_ch_out[-1], self.num_cls)
        return tail, se, body, head, score

    def forward(self, data):

        # for i in range()
        tail0 = self.tail[0](data)
        tail1 = self.tail[1](tail0)
        se = self.se[0](tail1)

        body0 = self.body[0](se)
        body1 = self.body[1](torch.cat((body0, tail0), dim=1))

        head0 = self.head[0](torch.cat((body1, data), dim=1))
        head1 = self.head[1](torch.cat((head0, body0), dim=1))

        score = self.score[0](torch.cat((head1, tail1), dim=1))
        score = self.score[1](score)
        return score


def make_blocks(num, ch_in, ch_out, res_blks, part):
    blocks = []
    is_down = True if part != 'body' else False
    sampling = nn.AvgPool2d(2, stride=2) if is_down else nn.Upsample(scale_factor=2)
    for i in range(num):
        k = int(round(ch_in[i]/ch_out[i])) if part == 'body' else 1
        block = []
        block.extend(make_res_block(ch_in[i], ch_out[i], res_blks[i], k=k, is_down=is_down))
        block.append(sampling)
        block = nn.Sequential(*block)
        blocks.append(block)
    return nn.ModuleList(blocks)


def make_res_block(ch_in, ch_out, num_res_blocks, is_down=False, k=1, dilation=1):
    layers = []

    if is_down:
        layers.append(FishBlock(ch_in, ch_out, stride=1))
    else:
        layers.append(FishBlock(ch_in, ch_out, mode='UR', dilation=dilation, k=k))

    for i in range(1, num_res_blocks):
        layers.append(FishBlock(ch_out, ch_out, stride=1, dilation=dilation))

    return layers


def make_se_block(ch_in, ch_out):
    bn = nn.BatchNorm2d(ch_in)
    conv_sq = nn.Conv2d(ch_in, ch_out//16, kernel_size=1)
    conv_ex = nn.Conv2d(ch_out//16, ch_out, kernel_size=1)
    relu = nn.ReLU(inplace=True)
    pool = nn.AdaptiveAvgPool2d(1)
    sigmoid = nn.Sigmoid()
    return nn.Sequential(bn, relu, pool, conv_sq, relu, conv_ex, sigmoid)


def make_score(ch_in, ch_out, has_pool=True):
    bn_in = nn.BatchNorm2d(ch_in)
    relu = nn.LeakyReLU(inplace=True)
    conv_trans = nn.Conv2d(ch_in, ch_in//2, kernel_size=1, bias=False)
    bn_out = nn.BatchNorm2d(ch_in//2)
    layers = nn.Sequential(bn_in, relu, conv_trans, bn_out, relu)
    if has_pool:
        fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in//2, ch_out, kernel_size=1, bias=True)
        )
    else:
        fc = nn.Conv2d(ch_in//2, ch_out, kernel_size=1, bias=True)
    return nn.Sequential(*[layers, fc])
