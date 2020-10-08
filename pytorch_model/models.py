import torch
from torch import nn
import torch.nn.functional as F
import math
from .utils_network import UnetConv3, UnetDsv3, UnetUp3_CT, UnetUp3


class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class unet_CT_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, is_dsv=False):
        super(unet_CT_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.is_dsv = is_dsv

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        if is_dsv:
            self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
            self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
            self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
            self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
            self.out1 = nn.Conv3d(n_classes*4, n_classes, 3, 1, 1, bias=False)
            self.out2 = nn.Conv3d(n_classes*4, 3, 3, 1, 1, bias=False)

        # final conv (without any concat)
        # self.final = nn.Conv3d(n_classes*4, n_classes, 1)
        else:
            self.out1 = nn.Conv3d(filters[0], n_classes, 3, 1, 1, bias=False)
            self.out2 = nn.Conv3d(filters[0], 3, 3, 1, 1, bias=False)


        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        if self.is_dsv:
            dsv4 = self.dsv4(up4)
            dsv3 = self.dsv3(up3)
            dsv2 = self.dsv2(up2)
            dsv1 = self.dsv1(up1)
            up1 = torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1)

        cnt = self.out1(up1)
        cnt = torch.sigmoid(cnt)
        sze = self.out2(up1)
        return cnt, sze

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class unet_CT_single_att_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(unet_CT_single_att_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1


class CenterNet3d(nn.Module):
    def __init__(self, num_class=1, outpooling=False):
        super(CenterNet3d, self).__init__()
        self.filter = [8, 16, 32, 64]
        self.ispooling = outpooling
        self.maxpooling = nn.MaxPool3d(2, 2)
        self.out1 = nn.Conv3d(self.filter[0], num_class, 3, 1, 1, bias=False)
        self.outpooling = nn.MaxPool3d(3, 1, 1)
        self.out2 = nn.Conv3d(self.filter[0], 3, 3, 1, 1, bias=False)
        self.conv1 = self.make_down_layer(1, self.filter[0])
        self.conv2 = self.make_down_layer(self.filter[0], self.filter[1])
        self.conv3 = self.make_down_layer(self.filter[1], self.filter[2])
        self.conv4 = self.make_down_layer(self.filter[2], self.filter[3])
        self.upSample1 = self.make_up_layer(self.filter[3], self.filter[3])
        self.upSample2 = self.make_up_layer(self.filter[3], self.filter[2])
        self.upSample3 = self.make_up_layer(self.filter[2], self.filter[1])
        self.upSample4 = self.make_up_layer(self.filter[1], self.filter[0])
        self.upConv1 = self.make_layer(self.filter[3] + self.filter[3] // 2, self.filter[3])
        self.upConv2 = self.make_layer(self.filter[2] + self.filter[2] // 2, self.filter[2])
        self.upConv3 = self.make_layer(self.filter[1] + self.filter[1] // 2, self.filter[1])
        self.upConv4 = self.make_layer(self.filter[0] + self.filter[0] // 2, self.filter[0])

    def make_down_layer(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels // 2, 3, 1, 1),
            nn.BatchNorm3d(output_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_channels // 2, output_channels, 3, 1, 1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )

    def make_up_layer(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(input_channels, output_channels // 2, 3, 1, 1),
            nn.BatchNorm3d(output_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_channels // 2, output_channels // 2, 3, 1, 1),
            nn.BatchNorm3d(output_channels // 2),
            nn.ReLU(inplace=True),
        )

    def make_layer(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, 1, 1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # 1, 128, 128, 128
        x1 = self.conv1(x)  # 8, 128, 128, 128

        x2 = self.maxpooling(x1)  # 8, 64, 64, 64
        x2 = self.conv2(x2)  # 16, 64, 64, 64

        x3 = self.maxpooling(x2)  # 16, 32, 32, 32
        x3 = self.conv3(x3)  # 32, 32, 32, 32

        x4 = self.maxpooling(x3)  # 32 ,16, 16, 16
        x4 = self.conv4(x4)  # 64, 16, 16, 16

        x5 = self.maxpooling(x4)  # 64, 8, 8, 8

        x5 = self.upSample1(x5)

        x5 = torch.cat([x5, x4], dim=1)
        x5 = self.upConv1(x5)

        x5 = self.upSample2(x5)

        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.upConv2(x5)

        x5 = self.upSample3(x5)
        x5 = torch.cat([x5, x2], dim=1)
        x5 = self.upConv3(x5)

        x5 = self.upSample4(x5)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.upConv4(x5)

        cnt_pred = self.out1(x5)
        if self.ispooling:
            cnt_pred = self.outpooling(cnt_pred)
        sze_pred = self.out2(x5)
        cnt_pred = F.sigmoid(cnt_pred)
        return cnt_pred, sze_pred


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()

    # num_pos = tf.reduce_sum(tf.cast(cnt_gt == 1, tf.float32))
    # # print ('num_pos:', num_pos)
    # neg_weights = math.pow(1 - cnt_gt, 4)
    # pos_weights = tf.ones_like(cnt_preds, dtype=tf.float32)
    # weights = tf.where(cnt_gt == 1, pos_weights, neg_weights)
    # inverse_preds = tf.where(cnt_gt == 1, cnt_preds, 1 - cnt_preds)
    #
    # loss = math.log(inverse_preds + 0.0001) * math.pow(1 - inverse_preds, 2) * weights
    # loss = tf.reduce_mean(loss)
    # loss = -loss / (num_pos * 1.0 + 1) * 1000000
    def forward(self, preds, targets):
        # print(cnt_pred, cnt_gt)
        # num_pos = torch.sum(cnt_gt == 1.0).type(torch.float32)
        # # print(num_pos)
        # neg_weights = torch.pow(1 - cnt_gt, 4).type(torch.float32)
        # pos_weights = torch.ones_like(cnt_pred, dtype=torch.float32)
        # weights = torch.where(cnt_gt == 1.0, pos_weights, neg_weights)
        # inverse_preds = torch.where(cnt_gt == 1.0, cnt_pred, 1 - cnt_pred)
        # # print(inverse_preds.shape, inverse_preds)
        # loss = torch.log(inverse_preds + 0.0001) * torch.pow(1 - inverse_preds, 2) * weights
        # loss = torch.sum(loss)
        # loss = -loss / (num_pos * 1.0 + 1)  # * 100000

        pos_inds = targets.eq(1).float()  # heatmap为1的部分是正样本
        neg_inds = targets.lt(1).float()  # 其他部分为负样本

        neg_weights = torch.pow(1 - targets, 4)  # 对应(1-Yxyc)^4

        loss = 0
        for pred in preds:  # 预测值
            # 约束在0-1之间
            pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
            pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
            neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
            num_pos = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if num_pos == 0:
                loss = loss - neg_loss  # 只有负样本
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
        return loss / len(preds)


class SizeLoss(nn.Module):
    def __init__(self):
        super(SizeLoss, self).__init__()

    # sze_gt = math_ops.cast(sze_gt, sze_preds.dtype)
    # mask = tf.where(sze_gt != (0, 0, 0), tf.ones_like(sze_gt), tf.zeros_like(sze_gt))
    # fg_num = math_ops.cast(tf.math.count_nonzero(mask), sze_preds.dtype) / 3.0
    # # print ('fg_num:',fg_num)
    # regr_loss = tf.reduce_sum(tf.abs(sze_gt - sze_preds) * mask) / fg_num
    def forward(self, sze_pred, sze_gt):
        mask = torch.where(sze_gt != 0, torch.ones_like(sze_gt), torch.zeros_like(sze_gt))
        fg_num = torch.sum(mask == 1) / 3.0
        loss = torch.sum(torch.abs(sze_gt - sze_pred) * mask) / fg_num
        return loss