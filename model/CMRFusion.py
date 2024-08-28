import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res import resblock
from model.Transformer import Restormer
from model.Mamba import SingleMambaBlock



class exblock(nn.Module):
    def __init__(self, in_channel, out_channel, num):
        super(exblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.TransformerBlock = nn.Sequential(
            *[Restormer(out_channel,5) for i in range(num)])
        self.MambaBlock = nn.Sequential(*[SingleMambaBlock(int(out_channel)) for i in range(num)])
        self.DenseBlock = nn.Sequential(*[resblock(out_channel, out_channel) for i in range(num)])
        self.dconv = nn.Conv2d(out_channel * 2, out_channel, 3, 1, 1)

    def forward(self, x):
        acctivate = nn.LeakyReLU()
        out = acctivate(self.conv1(x))
        out = self.TransformerBlock(out)
        outglobal = self.MambaBlock(out)
        outlocal = self.DenseBlock(out)
        out = self.dconv(torch.cat([outglobal, outlocal], 1))
        return out


class Dexblock(nn.Module):
    def __init__(self, in_channel, out_channel, num):
        super(Dexblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.TransformerBlock = nn.Sequential(
            *[Restormer(out_channel, 5) for i in range(num)])
        self.MambaBlock = nn.Sequential(*[SingleMambaBlock(int(out_channel)) for i in range(num)])
        self.DenseBlock = nn.Sequential(*[resblock(out_channel, out_channel) for i in range(num)])
        self.dconv = nn.Conv2d(out_channel * 2, out_channel, 3, 2, 1)
        self.msfe = MSFE(out_channel,out_channel)
    def forward(self, x):
        acctivate = nn.LeakyReLU()
        out = acctivate(self.conv1(x))
        out = self.TransformerBlock(out)
        outglobal = self.MambaBlock(out)
        outlocal = self.DenseBlock(out)
        out = self.dconv(torch.cat([outglobal, outlocal], 1))
        out = self.msfe(out)
        return out


class upxblock(nn.Module):
    def __init__(self, in_channel, out_channel, num):
        super(upxblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.TransformerBlock = nn.Sequential(
            *[Restormer(out_channel, 5) for i in range(num)])
        self.MambaBlock = nn.Sequential(*[SingleMambaBlock(int(out_channel)) for i in range(num)])
        self.DenseBlock = nn.Sequential(*[resblock(out_channel, out_channel) for i in range(num)])
        self.conv2 = nn.Conv2d(out_channel * 2, out_channel, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

    def forward(self, x):
        acctivate = nn.LeakyReLU()
        out = acctivate(self.conv1(x))
        out = self.TransformerBlock(out)
        outglobal = self.MambaBlock(out)
        outlocal = self.DenseBlock(out)
        out = self.conv2(torch.cat([outglobal, outlocal], 1))
        out = self.up(out)
        return out

class MSFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSFE, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.conv2 = nn.Conv2d(in_channel,out_channel,1,1,0)
        self.conv3 = nn.Conv2d(in_channel,out_channel,5,1,2)
        self.conv4 = nn.Conv2d(in_channel,out_channel,7,1,3)
        self.conv5 = nn.Conv2d(out_channel*4,out_channel,1,1,0)
    def forward(self,feat):
        out = torch.cat([self.conv1(feat) ,self.conv2(feat),self.conv3(feat),self.conv4(feat)],1)
        out = self.conv5(out)
        return out

class MKFusion(nn.Module):
    def __init__(self,numblock=[2, 3, 3, 4]):
        super(MKFusion, self).__init__()
        self.num_layers = 30
        self.conv1 = nn.Sequential(
            exblock(1, self.num_layers, numblock[0]),
            nn.PReLU(),
        )
        self.down_conv1 = nn.Sequential(
            Dexblock(self.num_layers, self.num_layers, numblock[1]),
            nn.PReLU(),
        )
        self.down_conv2 = nn.Sequential(
            Dexblock(self.num_layers, self.num_layers, numblock[2]),
            nn.PReLU(),
        )
        self.down_conv3 = nn.Sequential(
            Dexblock(self.num_layers, self.num_layers, numblock[3]),
            nn.PReLU(),
        )

        self.conv_t1 = nn.Sequential(
            exblock(1, self.num_layers, numblock[0]),
            # nn.BatchNorm2d(self.num_layers),
            nn.PReLU(),
        )
        self.down_conv_t1 = nn.Sequential(
            Dexblock(self.num_layers, self.num_layers, numblock[1]),
            nn.PReLU(),
        )
        self.down_conv_t2 = nn.Sequential(
            Dexblock(self.num_layers, self.num_layers, numblock[2]),
            nn.PReLU(),
        )
        self.down_conv_t3 = nn.Sequential(
            Dexblock(self.num_layers, self.num_layers, numblock[3]),
            nn.PReLU(),
        )

        self.GEFM1 = CDIM(self.num_layers, self.num_layers)
        self.GEFM2 = CDIM(self.num_layers, self.num_layers)
        self.GEFM3 = CDIM(self.num_layers, self.num_layers)
        self.GEFM4 = CDIM(self.num_layers, self.num_layers)

        self.up_conv3 = nn.Sequential(
            upxblock(self.num_layers * 2, self.num_layers, numblock[2]),
            nn.PReLU(),
        )
        self.up_conv2 = nn.Sequential(
            upxblock(self.num_layers * 3, self.num_layers, numblock[1]),
            nn.PReLU(),
        )
        self.up_conv1 = nn.Sequential(
            upxblock(self.num_layers * 3, self.num_layers, numblock[0]),
            nn.PReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(self.num_layers * 3, self.num_layers * 2, kernel_size=13, stride=1, padding=6),
            nn.Sequential(
            *[Restormer(self.num_layers*2, 5) for i in range(2)]),
            nn.PReLU(),
            nn.Conv2d(self.num_layers * 2, self.num_layers, kernel_size=7, stride=1, padding=3),
            nn.Sequential(
                *[Restormer(self.num_layers, 5) for i in range(2)]),
            nn.PReLU(),
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=5, stride=1, padding=2),
            nn.Sequential(
                *[Restormer(self.num_layers, 5) for i in range(2)]),
            nn.PReLU(),
            nn.Conv2d(self.num_layers, self.num_layers // 2, kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                *[Restormer(self.num_layers // 2, 5) for i in range(2)]),
            nn.PReLU(),
            nn.Conv2d(self.num_layers // 2, self.num_layers // 2, kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                *[Restormer(self.num_layers//2, 5) for i in range(2)]),
            nn.PReLU(),
            nn.Conv2d(self.num_layers // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )



    def encoder(self, thermal, rgb):
        rgb1 = self.conv1(rgb)
        rgb2 = self.down_conv1(rgb1)
        rgb3 = self.down_conv2(rgb2)
        rgb4 = self.down_conv3(rgb3)

        thermal1 = self.conv_t1(thermal)
        thermal2 = self.down_conv_t1(thermal1)
        thermal3 = self.down_conv_t2(thermal2)
        thermal4 = self.down_conv_t3(thermal3)

        return rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4

    def cross_modal_fusion(self, rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4):
        sem1 = self.GEFM1(rgb1, thermal1)
        sem2 = self.GEFM2(rgb2, thermal2)
        sem3 = self.GEFM3(rgb3, thermal3)
        sem4 = self.GEFM4(rgb4, thermal4)

        return sem1, sem2, sem3, sem4

    def decoder_fusion(self, sem1, sem2, sem3, sem4, rgb1, rgb2, rgb3, rgb4):
        fuse_de3 = self.up_conv3(torch.cat((rgb4, sem4), 1))
        fuse_de2 = self.up_conv2(torch.cat((fuse_de3, rgb3, sem3), 1))
        fuse_de1 = self.up_conv1(torch.cat((fuse_de2, rgb2, sem2), 1))
        fused_img = self.fusion(torch.cat((fuse_de1, rgb1, sem1), 1))

        return fused_img


    def clip_norm(im):
        DEV = im.device
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        im_re = F.interpolate(im.repeat(1, 3, 1, 1) if im.shape[1] == 1 else im, size=224, mode='bilinear',
                              align_corners=False)
        im_norm = (im_re - mean) / std
        return im_norm

    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth
        # v = self.get_image_feature(v).to(rgb.dtype)
        #####################Shared encoder#####################

        rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4 = self.encoder(thermal, rgb)
        sem1, sem2, sem3, sem4 = self.cross_modal_fusion(rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4)
        # sem1, sem2, sem3, sem4 = self.fuse(sem1, sem2, sem3, sem4, v)
        #####################Fusion decoder#####################

        fused_img = self.decoder_fusion(sem1, sem2, sem3, sem4, rgb1, rgb2, rgb3, rgb4)

        #####################Segmentation decoder#####################

        return fused_img
    @torch.no_grad()
    def get_image_feature(self, img):
        # print(img.size())
        image_features = self.model_clip.encode_image(img)
        return image_features







class BBasicConv2d(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x) * x_source + x_source


class CDIM(nn.Module):
    def __init__(self, in_C, out_C, size=32):
        super(CDIM, self).__init__()

        self.size = size

        self.RGB_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_Q = BBasicConv2d(in_C, out_C, 3, 1, 1)

        self.INF_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_Q = BBasicConv2d(out_C, out_C, 3, 1, 1)

        self.REDUCE = BBasicConv2d(out_C * 4, out_C, 3, 1, 1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.RGB_SPA_ATT = SpatialAttention()
        self.INF_SPA_ATT = SpatialAttention()

        self.SEC_REDUCE = BBasicConv2d(out_C * 3, out_C, 3, 1, 1)

    def forward(self, x, y):
        m_batchsize, c, h, w = x.shape

        # Resize inputs
        x_re = F.interpolate(x, size=(self.size, self.size), mode='bicubic')
        y_re = F.interpolate(y, size=(self.size, self.size), mode='bicubic')

        # Compute Q, K, V for both RGB and INF
        RGB_Q, RGB_K, RGB_V = self.compute_qkv(x_re, self.RGB_Q, self.RGB_K, self.RGB_V)
        INF_Q, INF_K, INF_V = self.compute_qkv(y_re, self.INF_Q, self.INF_K, self.INF_V)

        # Sum of RGB and INF V
        DUAL_V = RGB_V + INF_V

        # Attention mechanisms
        RGB_refine = self.attention_mechanism(RGB_Q, RGB_K, DUAL_V, x, self.gamma1, h, w)
        INF_refine = self.attention_mechanism(INF_Q, INF_K, DUAL_V, y, self.gamma2, h, w)
        RGB_INF_refine = self.attention_mechanism(RGB_Q, INF_K, RGB_V, y, self.gamma3, h, w)
        INF_RGB_refine = self.attention_mechanism(INF_Q, RGB_K, INF_V, x, self.gamma4, h, w)

        # Global attention
        GLOBAL_ATT = self.REDUCE(torch.cat((RGB_refine, INF_refine, RGB_INF_refine, INF_RGB_refine), 1))

        # Spatial attention
        RGB_SPA_ATT = self.RGB_SPA_ATT(x)
        INF_SPA_ATT = self.INF_SPA_ATT(y)

        # Output
        out = self.SEC_REDUCE(torch.cat([GLOBAL_ATT, INF_SPA_ATT, RGB_SPA_ATT], dim=1))

        return out

    def compute_qkv(self, x, Q_layer, K_layer, V_layer):
        Q = Q_layer(x).view(x.size(0), -1, self.size * self.size)
        K = K_layer(x).view(x.size(0), -1, self.size * self.size).permute(0, 2, 1)
        V = V_layer(x).view(x.size(0), -1, self.size * self.size)
        return Q, K, V

    def attention_mechanism(self, Q, K, V, original, gamma, h, w):
        mask = torch.bmm(K, Q)
        mask = self.softmax(mask)
        refine = torch.bmm(V, mask.permute(0, 2, 1))
        refine = refine.view(original.size(0), -1, self.size, self.size)
        refine = gamma * refine
        refine = F.interpolate(refine, size=(h, w), mode='bicubic') + original
        return refine


if __name__ == '__main__':
    model = MKFusion(9, 3, 3).cuda()
    img1 = torch.randn((1, 3, 128, 128)).cuda()
    img2 = torch.randn((1, 3, 128, 128)).cuda()
    out = model(img1, img2)
    print(out.shape)
