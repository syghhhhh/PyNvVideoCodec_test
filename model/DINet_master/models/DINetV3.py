import time
import torch
from torch import nn
import torch.nn.functional as F
from ..sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from ..sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d


def make_coordinate_grid_3d(spatial_size, type):
    """
        generate 3D coordinate grid
    """
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed, zz


class ResBlock1d(nn.Module):
    """
        basic block
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features, out_features, 1)
        self.norm1 = BatchNorm1d(in_features)
        self.norm2 = BatchNorm1d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class ResBlock2d(nn.Module):
    """
            basic block
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class UpBlock2d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock1d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=2)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class DownBlock2d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock1d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock2d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class AdaAT(nn.Module):
    '''
       AdaAT operator
    '''

    def __init__(self, para_ch, feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
            nn.Linear(para_ch, feature_ch),
            nn.Sigmoid()
        )
        self.rotation = nn.Sequential(
            nn.Linear(para_ch, feature_ch),
            nn.Tanh()
        )
        self.translation = nn.Sequential(
            nn.Linear(para_ch, 2 * feature_ch),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, para_code):
        # feature_map:[bs, 256, 104, 80], para_code:[bs, 256]
        batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        para_code = self.commn_linear(para_code)  # [bs, 256]->[bs, 256]
        scale = self.scale(para_code).unsqueeze(-1) * 2  # [bs, 256]->[bs, 256, 1]
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159  # [bs, 256]->[bs, 256, 1]
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        return trans_feature


class DINet(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):  # 3,15,29
        super(DINet, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 416, 320] -> [bs, 64, 416, 320]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 416, 320] -> [bs, 128, 208, 160]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 208, 160] -> [bs, 256, 104, 80]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 416, 320] -> [bs, 64, 416, 320]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 416, 320] -> [bs, 128, 208, 160]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 208, 160] -> [bs, 256, 104, 80]
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 104, 80] -> [bs, 128, 104, 80]
            SameBlock2d(128, 128, kernel_size=11, padding=5),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            SameBlock2d(128, 128, kernel_size=11, padding=5),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 52, 40]
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),  # [bs, 128, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=7, padding=3),  # [bs, 128, 52, 40] -> [bs, 128, 52, 40]
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 26, 20]
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 13, 10]
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]

        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),  # [bs, 29, 5] -> [bs, 128, 5]
            ResBlock1d(128, 128, 3, 1),  # [bs, 128, 5] -> [bs, 128, 5]
            DownBlock1d(128, 128, 3, 1),  # [bs, 128, 5] -> [bs, 128, 3]
            ResBlock1d(128, 128, 3, 1),  # [bs, 128, 3] -> [bs, 128, 3]
            DownBlock1d(128, 128, 3, 1),  # [bs, 128, 3] -> [bs, 128, 3]
            SameBlock1d(128, 128, kernel_size=3, padding=1)  # [bs, 128, 3] -> [bs, 128, 2]
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 208, 160] -> [bs, 128, 208, 160]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 208, 160] -> [bs, 128, 416, 320]
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 416, 320] -> [bs, 3, 416, 320]
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 416, 320]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 104, 80]

        ## reference image encoder
        # ref_img: [bs, 15, 416, 320]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 104, 80]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 7, 5]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        ## audio encoder
        # audio_feature: [bs, 29, 5]
        audio_para = self.audio_encoder(audio_feature)
        # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 128, 2]
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 128]

        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para, audio_para], 1)
        # print(f'trans_para.shape:{trans_para.shape}')  # [bs, 256]

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]

        ## feature decoder
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 512, 104, 80]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 416, 320]
        return out


# img_para仅读取ref_in_feature的信息
class DINetV3(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):  # 3,15,29
        super(DINetV3, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]

        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),  # [bs, 2048, 5] -> [bs, 128, 5]
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),  # [bs, 128, 5] -> [bs, 128, 3]
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),  # [bs, 128, 3] -> [bs, 128, 2]
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 208, 160] -> [bs, 128, 208, 160]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 208, 160] -> [bs, 128, 416, 320]
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 416, 320] -> [bs, 3, 416, 320]
            SameBlock2d(3, 3, kernel_size=7, padding=3),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            SameBlock2d(3, 3, kernel_size=3, padding=1),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            nn.Conv2d(3, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(ref_in_feature)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 4, 3]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        ## audio encoder
        # audio_feature: [bs, 2048, 5]
        audio_para = self.audio_encoder(audio_feature)
        # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 128, 2]
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 128]

        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para, audio_para], 1)
        # print(f'trans_para.shape:{trans_para.shape}')  # [bs, 256]

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]

        ## feature decoder
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 512, 104, 80]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 416, 320]
        return out


# 调整trans_conv,audio_encoder;先audio_para生成新口型,再img_para调整生成新姿态
class DINetV3p1(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):  # 3,15,2048
        super(DINetV3p1, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.audio_encoder = nn.Sequential(
            # [bs, 2048, 5] -> [bs, 1024, 5]
            SameBlock1d(audio_channel, 1024, kernel_size=5, padding=2),
            ResBlock1d(1024, 1024, 3, 1),
            # [bs, 1024, 5] -> [bs, 512, 5]
            ResBlock1d(1024, 512, 3, 1),
            ResBlock1d(512, 512, 3, 1),
            # [bs, 512, 5] -> [bs, 512, 3]
            DownBlock1d(512, 512, 3, 1),
            ResBlock1d(512, 512, 3, 1),
            # [bs, 512, 3] -> [bs, 256, 3]
            ResBlock1d(512, 256, 3, 1),
            ResBlock1d(256, 256, 3, 1),
            # [bs, 256, 3] -> [bs, 256, 2]
            DownBlock1d(256, 256, 3, 1),
            SameBlock1d(256, 256, kernel_size=3, padding=1)
        )
        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.adaAT128 = AdaAT(128, 128)
        self.out_source = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            SameBlock2d(3, 3, kernel_size=7, padding=3),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            SameBlock2d(3, 3, kernel_size=3, padding=1),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            nn.Conv2d(3, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 4, 3]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        ## audio encoder
        # audio_feature: [bs, 2048, 5]
        audio_para = self.audio_encoder(audio_feature)
        # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 256, 2]
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 256]

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT256(ref_trans_feature, audio_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]
        ref_trans_feature = self.adaAT128(ref_trans_feature, img_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]
        ref_trans_feature = self.appearance_conv_list[2](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([self.out_source(source_in_feature), ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


# 原版的基础上,audio_encoder部分使用transformer
class DINetV3p2(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):  # 3,15,29
        super(DINetV3p2, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )
        self.audio_feature_map = nn.Linear(audio_channel, 128)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )
        appearance_conv_list = []
        for _ in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            SameBlock2d(3, 3, kernel_size=7, padding=3),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            SameBlock2d(3, 3, kernel_size=3, padding=1),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            nn.Conv2d(3, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 4, 3]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        # ## audio encoder
        # # audio_feature: [bs, 2048, 5]
        # audio_para = self.audio_encoder(audio_feature)
        # # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 256, 2]
        # audio_para = self.global_avg1d(audio_para).squeeze(2)
        # # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 256]

        # audio_feature = audio_feature.transpose(2, 3).squeeze(1)  # (B, 1, 2048, 5) -> (B, 5, 2048)
        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, 5) -> (B, 5, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, 5, 2048) -> (B, 5, 128)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, 5, 128) -> (B, 128, 5)
        audio_para = self.global_avg1d(audio_para).squeeze(2)  # (B, 128, 5) -> (B, 128, 1) -> (B, 128)

        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para, audio_para], 1)
        # print(f'trans_para.shape:{trans_para.shape}')  # [bs, 256]

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 512, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


# 3p1的结构中把audio_encoder改成3p2中的transformer
class DINetV3p3(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):  # 3,15,2048
        super(DINetV3p3, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        # self.audio_encoder = nn.Sequential(
        #     # [bs, 2048, 5] -> [bs, 1024, 5]
        #     SameBlock1d(audio_channel, 1024, kernel_size=5, padding=2),
        #     ResBlock1d(1024, 1024, 3, 1),
        #     # [bs, 1024, 5] -> [bs, 512, 5]
        #     ResBlock1d(1024, 512, 3, 1),
        #     ResBlock1d(512, 512, 3, 1),
        #     # [bs, 512, 5] -> [bs, 512, 3]
        #     DownBlock1d(512, 512, 3, 1),
        #     ResBlock1d(512, 512, 3, 1),
        #     # [bs, 512, 3] -> [bs, 256, 3]
        #     ResBlock1d(512, 256, 3, 1),
        #     ResBlock1d(256, 256, 3, 1),
        #     # [bs, 256, 3] -> [bs, 256, 2]
        #     DownBlock1d(256, 256, 3, 1),
        #     SameBlock1d(256, 256, kernel_size=3, padding=1)
        # )
        self.audio_feature_map = nn.Linear(audio_channel, 256)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )

        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.adaAT128 = AdaAT(128, 128)
        self.out_source = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            SameBlock2d(3, 3, kernel_size=7, padding=3),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            SameBlock2d(3, 3, kernel_size=3, padding=1),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            nn.Conv2d(3, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 4, 3]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        ## audio encoder
        # # audio_feature: [bs, 2048, 5]
        # audio_para = self.audio_encoder(audio_feature)
        # # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 256, 2]
        # audio_para = self.global_avg1d(audio_para).squeeze(2)
        # # print(f'audio_para.shape:{audio_para.shape}')  # [bs, 256]

        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, 5) -> (B, 5, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, 5, 2048) -> (B, 5, 256)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, 5, 256) -> (B, 256, 5)
        audio_para = self.global_avg1d(audio_para).squeeze(2)  # (B, 256, 5) -> (B, 256, 1) -> (B, 256)

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT256(ref_trans_feature, audio_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]
        ref_trans_feature = self.adaAT128(ref_trans_feature, img_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]
        ref_trans_feature = self.appearance_conv_list[2](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([self.out_source(source_in_feature), ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


class MyModel(nn.Module):
    def __init__(self, len_audio):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(len_audio, len_audio, bias=False)

    def forward(self, x):
        weights = F.softmax(self.linear(torch.ones_like(x)), dim=2)
        return (x * weights).sum(dim=2, keepdim=True)


# 3p3的基础上,音频特征提取把global_avg1d替换成线性层,可以调整音频不同时间点的权重
class DINetV3p4(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, audio_len):  # 3,15,2048
        super(DINetV3p4, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )

        self.audio_feature_map = nn.Linear(audio_channel, 256)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )

        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.adaAT128 = AdaAT(128, 128)
        self.out_source = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            SameBlock2d(3, 3, kernel_size=7, padding=3),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            SameBlock2d(3, 3, kernel_size=3, padding=1),
            ResBlock2d(3, 3, kernel_size=3, padding=1),
            nn.Conv2d(3, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        # self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.audio_global_avg1d = MyModel(audio_len)


    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 4, 3]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, 5) -> (B, 5, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, 5, 2048) -> (B, 5, 256)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, 5, 256) -> (B, 256, 5)
        # audio_para = self.global_avg1d(audio_para).squeeze(2)  # (B, 256, 15) -> (B, 256, 1) -> (B, 256)
        audio_para = self.audio_global_avg1d(audio_para).squeeze(2)  # (B, 256, 15) -> (B, 256, 1) -> (B, 256)
        # print(audio_para.shape)

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT256(ref_trans_feature, audio_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]
        ref_trans_feature = self.adaAT128(ref_trans_feature, img_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]
        ref_trans_feature = self.appearance_conv_list[2](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([self.out_source(source_in_feature), ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


# 同v4p2,输入的source_channel为3,不带alpha信息
class DINetV3p5(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, audio_len):  # 4,20,2048,7
        super(DINetV3p5, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 4, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )

        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 20, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )

        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )

        self.audio_feature_map = nn.Linear(audio_channel, 128)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, source_channel, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            # SameBlock2d(source_channel, source_channel, kernel_size=7, padding=3),
            # ResBlock2d(source_channel, source_channel, kernel_size=3, padding=1),
            # SameBlock2d(source_channel, source_channel, kernel_size=3, padding=1),
            # ResBlock2d(source_channel, source_channel, kernel_size=3, padding=1),
            # nn.Conv2d(source_channel, source_channel, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        # self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.audio_global_avg1d = MyModel(audio_len)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 7, 5]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        ## audio encoder
        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, audio_len) -> (B, audio_len, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, audio_len, 2048) -> (B, audio_len, 128)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, audio_len, 128) -> (B, 128, audio_len)
        audio_para = self.audio_global_avg1d(audio_para).squeeze(2)  # (B, 128, audio_len) -> (B, 128, 1) -> (B, 128)

        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para, audio_para], 1)
        # print(f'trans_para.shape:{trans_para.shape}')  # [bs, 256]

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT256(ref_trans_feature, trans_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(Encoder, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pooling else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
                                      nn.ReLU(inplace=True))
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class MaskNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        super(MaskNet, self).__init__()
        self.inc = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.enc1 = Encoder(16, 32, pooling=True)
        self.enc2 = Encoder(32, 64, pooling=True)
        self.enc3 = Encoder(64, 128, pooling=True)
        self.enc4 = Encoder(128, 256, pooling=True)

        self.dec4 = Decoder(256, 128)
        self.dec3 = Decoder(128, 64)
        self.dec2 = Decoder(64, 32)
        self.dec1 = Decoder(32, 16)

        # self.final = nn.Conv2d(16, out_channels, kernel_size=1)
        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        enc1 = self.enc1(x1)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        dec1 = self.dec1(dec2, x1)

        final = self.final(dec1)

        return final


# 3p3基础上删掉第二次的AdaAT
class DINetV4p1(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):  # 3,15,2048
        super(DINetV4p1, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )
        self.audio_feature_map = nn.Linear(audio_channel, 256)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )

        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, source_channel, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            nn.Sigmoid()
        )
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, 5) -> (B, 5, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, 5, 2048) -> (B, 5, 256)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, 5, 256) -> (B, 256, 5)
        audio_para = self.global_avg1d(audio_para).squeeze(2)  # (B, 256, 5) -> (B, 256, 1) -> (B, 256)

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT256(ref_trans_feature, audio_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


class DINetV4p2(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, audio_len):  # 4,20,2048,7
        super(DINetV4p2, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 4, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1)  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )

        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 20, 208, 160] -> [bs, 64, 208, 160]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 208, 160] -> [bs, 128, 104, 80]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 256, 52, 40]
        )

        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 26, 20]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 26, 20] -> [bs, 128, 13, 10]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 13, 10] -> [bs, 128, 7, 5]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 7, 5] -> [bs, 128, 4, 3]
        )

        self.audio_feature_map = nn.Linear(audio_channel, 128)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 52, 40] -> [bs, 128, 52, 40]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 52, 40] -> [bs, 128, 104, 80]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 104, 80] -> [bs, 128, 104, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 104, 80] -> [bs, 128, 208, 160]
            nn.Conv2d(128, source_channel, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 208, 160] -> [bs, 3, 208, 160]
            # SameBlock2d(source_channel, source_channel, kernel_size=7, padding=3),
            # ResBlock2d(source_channel, source_channel, kernel_size=3, padding=1),
            # SameBlock2d(source_channel, source_channel, kernel_size=3, padding=1),
            # ResBlock2d(source_channel, source_channel, kernel_size=3, padding=1),
            # nn.Conv2d(source_channel, source_channel, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        # self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.audio_global_avg1d = MyModel(audio_len)

    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        # source_img: [bs, 3, 208, 160]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 52, 40]

        ## reference image encoder
        # ref_img: [bs, 15, 208, 160]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 52, 40]

        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 7, 5]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        ## audio encoder
        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, audio_len) -> (B, audio_len, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, audio_len, 2048) -> (B, audio_len, 128)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, audio_len, 128) -> (B, 128, audio_len)
        audio_para = self.audio_global_avg1d(audio_para).squeeze(2)  # (B, 128, audio_len) -> (B, 128, 1) -> (B, 128)

        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para, audio_para], 1)
        # print(f'trans_para.shape:{trans_para.shape}')  # [bs, 256]

        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.adaAT256(ref_trans_feature, trans_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 52, 40]

        ## feature decoder
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 52, 40]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 208, 160]
        return out


class DINetV4p3(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, audio_len=7):  # 4,20,2048,7
        super(DINetV4p3, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),  # [bs, 3, 320, 320] -> [bs, 64, 320, 320]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 320, 320] -> [bs, 128, 160, 160]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 160, 160] -> [bs, 256, 80, 80]
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),  # [bs, 15, 320, 320] -> [bs, 64, 320, 320]
            DownBlock2d(64, 128, kernel_size=3, padding=1),  # [bs, 64, 320, 320] -> [bs, 128, 160, 160]
            DownBlock2d(128, 256, kernel_size=3, padding=1),  # [bs, 128, 160, 160] -> [bs, 256, 80, 80]
        )

        self.audio_feature_map = nn.Linear(audio_channel, 256)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_encoder = nn.TransformerEncoder(self.audio_transformer_encoder_layer, num_layers=1)
        self.audio_transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=2, batch_first=True)
        self.audio_transformer_decoder = nn.TransformerDecoder(self.audio_transformer_decoder_layer, num_layers=1)
        self.audio_output = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Tanh(),
        )

        self.trans_conv = nn.Sequential(
            # 40→20
            SameBlock2d(512, 128, kernel_size=3, padding=1),  # [bs, 512, 80, 80] -> [bs, 128, 80, 80]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 80, 80] -> [bs, 128, 40, 40]
            # 20→10
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 40, 40] -> [bs, 128, 20, 20]
            # 10→5
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 20, 20] -> [bs, 128, 10, 10]
            # 5→3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 10, 10] -> [bs, 128, 5, 5]
        )

        appearance_conv_list = [
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 256, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(256, 256, 3, 1),
                ResBlock2d(256, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            ),
            nn.Sequential(
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
                ResBlock2d(128, 128, 3, 1),
            )
        ]
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT256 = AdaAT(256, 256)
        self.adaAT128 = AdaAT(128, 128)
        self.out_source = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 80, 80] -> [bs, 128, 80, 80]
            SameBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Sequential(
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # [bs, 256, 80, 80] -> [bs, 128, 80, 80]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 80, 80] -> [bs, 128, 160, 160]
            ResBlock2d(128, 128, 3, 1),  # [bs, 128, 160, 160] -> [bs, 128, 160, 160]
            UpBlock2d(128, 128, kernel_size=3, padding=1),  # [bs, 128, 160, 160] -> [bs, 128, 320, 320]
            nn.Conv2d(128, source_channel, kernel_size=(7, 7), padding=(3, 3)),  # [bs, 128, 320, 320] -> [bs, 3, 320, 320]
            # SameBlock2d(3, 3, kernel_size=7, padding=3),
            # ResBlock2d(3, 3, kernel_size=3, padding=1),
            # SameBlock2d(3, 3, kernel_size=3, padding=1),
            # ResBlock2d(3, 3, kernel_size=3, padding=1),
            # nn.Conv2d(3, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        # self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.audio_global_avg1d = MyModel(audio_len)


    def forward(self, source_img, ref_img, audio_feature):
        # source image encoder
        # source_img: [bs, 3, 320, 320]
        source_in_feature = self.source_in_conv(source_img)
        # print(f'source_in_feature.shape:{source_in_feature.shape}')  # [bs, 256, 104, 80]

        # reference image encoder
        # ref_img: [bs, 15, 416, 320]
        ref_in_feature = self.ref_in_conv(ref_img)
        # print(f'ref_in_feature.shape:{ref_in_feature.shape}')  # [bs, 256, 104, 80]

        # alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128, 4, 3]
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        # print(f'img_para.shape:{img_para.shape}')  # [bs, 128]

        audio_feature = audio_feature.transpose(1, 2)  # (B, 2048, 5) -> (B, 5, 2048)
        # print(audio_feature.shape)
        audio_feature = self.audio_feature_map(audio_feature)  # (B, 5, 2048) -> (B, 5, 256)
        # print(audio_feature.shape)
        transformer_encoder_output = self.audio_transformer_encoder(audio_feature)
        # print(transformer_encoder_output.shape)
        transformer_decoder_output = self.audio_transformer_decoder(tgt=transformer_encoder_output, memory=audio_feature)
        # print(transformer_decoder_output.shape)
        audio_para = self.audio_output(transformer_decoder_output).transpose(1, 2)  # (B, 5, 256) -> (B, 256, 5)
        # audio_para = self.global_avg1d(audio_para).squeeze(2)  # (B, 256, 15) -> (B, 256, 1) -> (B, 256)
        audio_para = self.audio_global_avg1d(audio_para).squeeze(2)  # (B, 256, 15) -> (B, 256, 1) -> (B, 256)
        # print(audio_para.shape)

        # use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]
        ref_trans_feature = self.adaAT256(ref_trans_feature, audio_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 256, 104, 80]
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 104, 80]
        ref_trans_feature = self.adaAT128(ref_trans_feature, img_para)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 104, 80]
        ref_trans_feature = self.appearance_conv_list[2](ref_trans_feature)
        # print(f'ref_trans_feature.shape:{ref_trans_feature.shape}')  # [bs, 128, 104, 80]

        # feature decoder
        merge_feature = torch.cat([self.out_source(source_in_feature), ref_trans_feature], 1)
        # print(f'merge_feature.shape:{merge_feature.shape}')  # [bs, 256, 104, 80]
        out = self.out_conv(merge_feature)
        # print(f'out.shape:{out.shape}')  # [bs, 3, 416, 320]
        return out


# 倒置残差网络
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=4):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 双卷积
class DoubleConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DoubleConvDW, self).__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride, use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1, use_res_connect=True, expand_ratio=2)
        )

    def forward(self, x):
        return self.double_conv(x)


# 输入
class InConvDw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConvDw, self).__init__()
        self.inconv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=1, use_res_connect=False, expand_ratio=2)
        )

    def forward(self, x):
        return self.inconv(x)


# 输出
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvDW(in_channels, out_channels, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], axis=1)
        return self.conv(x)


class AudioConvHubert2(nn.Module):
    def __init__(self):
        super(AudioConvHubert2, self).__init__()
        # ch = [16, 32, 64, 128, 256]   # if you want to run this model on a mobile device, use this.
        ch = [32, 64, 128, 256, 512, 1024]
        self.conv1 = InvertedResidual(32, ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()
        self.conv6 = InvertedResidual(ch[4], ch[5], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[5], ch[5], stride=1, use_res_connect=True, expand_ratio=2)

    def forward(self, x):
        x = self.conv1(x)  # [bs, 64, 32, 32]
        x = self.conv2(x)  # [bs, 128, 32, 32]
        x = self.relu(self.bn3(self.conv3(x)))  # [bs, 256, 16, 16]
        x = self.conv4(x)  # [bs, 256, 16, 16]
        x = self.relu(self.bn5(self.conv5(x)))  # [bs, 512, 10, 10]
        x = self.conv6(x)  # [bs, 1024, 10, 10]
        x = self.conv7(x)  # [bs, 1024, 10, 10]
        return x


class DINetV4p4(nn.Module):
    def __init__(self, n_channels=8):
        super(DINetV4p4, self).__init__()
        self.n_channels = n_channels  # BGRA*2 = 8
        self.out_channels = n_channels // 2  # 4
        # ch = [16, 32, 64, 128, 256]  # if you want to run this model on a mobile device, use this.
        ch = [32, 64, 128, 256, 512, 1024]
        self.audio_model = AudioConvHubert2()

        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[5] * 2, ch[5], stride=1),
            DoubleConvDW(ch[5], ch[4], stride=1)
        )

        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])
        self.down5 = Down(ch[4], ch[5])

        self.up1 = Up(ch[5], ch[4] // 2)
        self.up2 = Up(ch[4], ch[3] // 2)
        self.up3 = Up(ch[3], ch[2] // 2)
        self.up4 = Up(ch[2], ch[1] // 2)
        self.up5 = Up(ch[1], ch[0])

        self.outc = OutConv(ch[0], self.out_channels)

    def forward(self, x1, x2, audio_feat):
        x = torch.cat([x1, x2], axis=1)  # [bs, 8, 320, 320]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # [bs, 256, 40, 40]
        x5 = self.down4(x4)  # [bs, 512, 20, 20]
        x6 = self.down5(x5)  # [bs, 1024, 10, 10]
        # print(x6.shape)
        audio_feat = self.audio_model(audio_feat)
        # print(audio_feat.shape)
        x6 = torch.cat([x6, audio_feat], axis=1)
        x6 = self.fuse_conv(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        out = self.outc(x)
        out = F.sigmoid(out)
        return out


if __name__ == "__main__":
    task = 1
    if task == 1:
        # 测试模型输出形状
        # model = DINetV3p1(3, 15, 2048)
        # model = DINetV3p3(3, 15, 2048)
        # model = DINetV3p4(3, 15, 2048, 15)
        model = DINetV4p1(3, 15, 2048)
        model.eval()
        test_source_img = torch.randn(2, 3, 416, 320)
        test_ref_img = torch.randn(2, 15, 416, 320)
        test_audio_feature = torch.randn(2, 2048, 15)
        with torch.no_grad():
            out = model(test_source_img, test_ref_img, test_audio_feature)
        print(out.shape)  # torch.Size([2, 3, 416, 320])
    elif task == 2:
        # 新建模型DINetv2,测试能否继承DINet除了out_conv之外的所有层的参数
        # model_v1 = DINet(3, 15, 2048)
        model_v1 = torch.load(r'H:\dinet_dataset_\training_model_weight1119\best_model\netG_epoch_98_loss_2.6093.pth')
        model_v2 = DINetV3(3, 15, 2048)
        model_v2_dict = model_v2.state_dict()
        # # 打印model_v2的参数是否需要更新
        # for k, v in model_v2.named_parameters():
        #     print(k, v.requires_grad)

        model_v1_dict = model_v1['state_dict']['net_g']
        model_v1_dict = {k: v for k, v in model_v1_dict.items() if k in model_v2_dict}
        model_v2_dict.update(model_v1_dict)
        model_v2.load_state_dict(model_v2_dict)
        # 冻结model_v1_dict的参数
        for k, v in model_v2.named_parameters():
            if k in model_v1_dict:
                v.requires_grad = False
        # 显示哪些层参数没有被更新
        for k, v in model_v2.named_parameters():
            print(f'{k}:{v.requires_grad}')
        model_v2.eval()
        test_source_img = torch.randn(1, 3, 208, 160)
        test_ref_img = torch.randn(1, 15, 208, 160)
        test_audio_feature = torch.randn(1, 2048, 5)
        with torch.no_grad():
            out = model_v2(test_source_img, test_ref_img, test_audio_feature)
        print(out.shape)
    elif task == 3:
        # 测试模型推理时间
        model = DINet(3, 15, 2048).cuda()
        model.eval()
        test_source_img = torch.randn(1, 3, 208, 160).cuda()
        test_ref_img = torch.randn(1, 15, 208, 160).cuda()
        test_audio_feature = torch.randn(1, 2048, 5).cuda()
        t0 = time.time()
        for i in range(100):
            with torch.no_grad():
                out = model(test_source_img, test_ref_img, test_audio_feature)
        t1 = time.time()
        print(f'base model spend time:{(t1 - t0) / 100}')  # 0.0077

        model = DINetV3p1(3, 15, 2048).cuda()
        model.eval()
        test_source_img = torch.randn(1, 3, 208, 160).cuda()
        test_ref_img = torch.randn(1, 15, 208, 160).cuda()
        test_audio_feature = torch.randn(1, 2048, 5).cuda()
        t0 = time.time()
        for i in range(100):
            with torch.no_grad():
                out = model(test_source_img, test_ref_img, test_audio_feature)
        t1 = time.time()
        print(f'new model spend time:{(t1 - t0) / 100}')  # 0.0079

        model = DINet(3, 15, 2048).cuda()
        model.eval()
        test_source_img = torch.randn(1, 3, 208, 160).cuda()
        test_ref_img = torch.randn(1, 15, 208, 160).cuda()
        test_audio_feature = torch.randn(1, 2048, 5).cuda()
        t0 = time.time()
        for i in range(100):
            with torch.no_grad():
                out = model(test_source_img, test_ref_img, test_audio_feature)
        t1 = time.time()
        print(f'base model spend time:{(t1 - t0) / 100}')  # 0.0058
