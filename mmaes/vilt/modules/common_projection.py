import torch
from torch.nn import Module, Conv1d, Parameter, Softmax, BatchNorm1d, Sequential, ReLU, Dropout, Linear


class SpaceProjection(Module):
    def __init__(self, in_dim):
        super(SpaceProjection, self).__init__()
        self.in_channel = in_dim
        self.q = Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.k = Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.v = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        query = self.q(x).permute(0, 2, 1)
        key = self.k(x)
        value = self.v(x)
        attmap = torch.bmm(query, key)
        score = self.softmax(attmap).permute(0, 2, 1)

        out = torch.bmm(value, score)
        out = self.gamma * out + x
        return out


class ChannelProjection(Module):
    def __init__(self, in_dim):
        super(ChannelProjection, self).__init__()
        self.in_channel = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        query = x
        key = x.permute(0, 2, 1)
        value = x
        attmap = torch.bmm(query, key)
        score = self.softmax(attmap)

        out = torch.bmm(score, value)
        out = self.gamma*out + x
        return out


class MissModalityAlign(Module):
    def __init__(self, in_channels, out_channels):
        super(MissModalityAlign, self).__init__()
        middle_channels = in_channels // 4
        self.conv1 = Sequential(Conv1d(in_channels, middle_channels, 3, padding=1, bias=False),
                                 BatchNorm1d(middle_channels),
                                 ReLU())

        self.conv2 = Sequential(Conv1d(in_channels, middle_channels, 3, padding=1, bias=False),
                                 BatchNorm1d(middle_channels),
                                 ReLU())

        self.sp = SpaceProjection(middle_channels)
        self.cp = ChannelProjection(middle_channels)
        self.conv3 = Sequential(Conv1d(middle_channels, out_channels, 3, padding=1, bias=False),
                                 BatchNorm1d(out_channels),
                                 ReLU())
        self.conv4 = Sequential(Conv1d(middle_channels, out_channels, 3, padding=1, bias=False),
                                 BatchNorm1d(out_channels),
                                 ReLU())
        self.conv5 = Sequential(Dropout(0.1, False), Conv1d(middle_channels, out_channels, 1))
        # self.w1 = Linear(out_channels, out_channels)
        # self.w2 = Linear(out_channels, out_channels)

    def forward(self, text_embeds, image_embeds):
        text_embeds = text_embeds.permute(0, 2, 1)
        feat1 = self.conv1(text_embeds)
        text_feat = self.cp(feat1)
        text_conv = self.conv3(text_feat).permute(0, 2, 1)

        image_embeds = image_embeds.permute(0, 2, 1)
        feat2 = self.conv2(image_embeds)
        image_feat = self.sp(feat2)
        image_conv = self.conv4(image_feat).permute(0, 2, 1)

        co_embeds = torch.cat([text_conv, image_conv], dim=1)

        return co_embeds
        # x = x.permute(0, 2, 1)
        # feat1 = self.conv1(x)
        # sa_feat = self.sp(feat1)
        # sa_conv = self.conv3(sa_feat)
        #
        # feat2 = self.conv2(x)
        # sc_feat = self.cp(feat2)
        # sc_conv = self.conv4(sc_feat)
        #
        # feat_sum = sa_conv + sc_conv
        #
        # sasc_output = self.conv5(feat_sum).permute(0, 2, 1)
        #
        # return sasc_output


# x = torch.randn(16, 741, 768)
# mma = MissModalityAlign(768, 768)
# x = mma(x)
# print(x.shape)
