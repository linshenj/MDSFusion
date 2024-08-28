import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import numbers
from mamba_simple import Mamba
import torch
from model.kan_convolutional.KANLinear import KANLinear

def L1_norm(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = torch.abs(narry_a)
    temp_abs_b = torch.abs(narry_b)
    _l1_a = torch.sum(temp_abs_a, dim=1)
    _l1_b = torch.sum(temp_abs_b, dim=1)

    _l1_a = torch.sum(_l1_a, dim=0)
    _l1_b = torch.sum(_l1_b, dim=0)
    with torch.no_grad():
        l1_a = _l1_a.detach()
        l1_b = _l1_b.detach()

    # caculate the map for source images
    mask_value = l1_a + l1_b
    # print("mask_value 的size",mask_value.size())

    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b
    # print("array_MASK_b 的size",array_MASK_b.size())
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            temp_matrix = array_MASK_a * narry_a[i, j, :, :] + array_MASK_b * narry_b[i, j, :, :]
            # print("temp_matrix 的size",temp_matrix.size())
            result.append(temp_matrix)

    result = torch.stack(result, dim=-1)

    result = result.reshape((dimension[0], dimension[1], dimension[2], dimension[3]))

    return result

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops
class SingleMambaBlock(nn.Module):
    def __init__(self, dim,norm_layer=nn.LayerNorm, patch_norm=True,):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            img_size=128, patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed = PatchUnEmbed(
            img_size=128, patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):

        _, _, h, w = ipt.shape
        residual = ipt
        residual = self.patch_embed(residual)
        # print(residual.size())
        x = self.norm(residual)
        out = self.encoder(x)
        out = self.patch_unembed(out, (h,w))
        return out+ipt



class M3(nn.Module):
    def __init__(self, dim):
        super(M3, self).__init__()
        self.multi_modal_mamba_block = Mamba(dim,bimamba_type="m3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.norm3 = LayerNorm(dim,'with_bias')

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,I1,fusion_resi,I2,fusion,test_h,test_w):
        fusion_resi = fusion + fusion_resi
        fusion = self.norm1(fusion_resi)
        I2 = self.norm2(I2)
        I1 = self.norm3(I1)

        global_f = self.multi_modal_mamba_block(self.norm1(fusion), extra_emb1=self.norm2(I2), extra_emb2=self.norm3(I1))

        B,HW,C = global_f.shape
        fusion = global_f.transpose(1, 2).view(B, C, test_h, test_w)
        fusion =  (self.dwconv(fusion)+fusion).flatten(2).transpose(1, 2)
        return fusion,fusion_resi


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_A(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class sample(nn.Module):
    def __init__(self, n_feat):
        super(sample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
class EncoderA(nn.Module):

    def __init__(self,inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4]):
        super(EncoderA, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[SingleMambaBlock(dim) for i in range(num_blocks[0])])
        self.down1_2 = sample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 1)) for i in range(num_blocks[1])])
        self.down2_3 = sample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 2)) for i in range(num_blocks[2])])
        self.down3_4 = sample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 3)) for i in range(num_blocks[3])])
    def forward(self, inp_img_A):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class EncoderB(nn.Module):

    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4]):
        super(EncoderB, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[SingleMambaBlock(dim) for i in range(num_blocks[0])])
        self.down1_2 = sample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 1)) for i in range(num_blocks[1])])
        self.down2_3 = sample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 2)) for i in range(num_blocks[2])])
        self.down3_4 = sample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 3)) for i in range(num_blocks[3])])

    def forward(self, inp_img_B):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B)

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.KAN = nn.Sequential(
            KANLinear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            KANLinear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.KAN(text_embed).unsqueeze(1).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta

        return x
class M_IF(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3,out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],bias = False):

        super(M_IF, self).__init__()

        self.model_clip = model_clip

        self.model_clip.eval()
        self.encoder_A = EncoderA(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks)
        self.encoder_B = EncoderB(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks)
        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)
        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)

        self.decoder_level4 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 3)) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim=dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 2)) for i in range(num_blocks[2])])

        self.feature_fusion_2 = Fusion_Embed(embed_dim=dim * 2 ** 1)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 1)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 1)) for i in range(num_blocks[1])])

        self.feature_fusion_1 = Fusion_Embed(embed_dim=dim)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 1)) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[SingleMambaBlock(int(dim * 2 ** 1)) for i in range(4)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img_A, inp_img_B,v):
        text_features = self.get_image_feature(v).to(inp_img_A.dtype)

        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(inp_img_A)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = self.encoder_B(inp_img_B)

        out_enc_level4_A, out_enc_level4_B = self.cross_attention(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.attention_spatial(out_enc_level4)

        out_enc_level4 = self.prompt_guidance_4(out_enc_level4, text_features)
        inp_dec_level4 = out_enc_level4

        out_dec_level4 = self.decoder_level4(inp_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)
        # print(inp_dec_level3.size())
        inp_dec_level3 = self.prompt_guidance_3(inp_dec_level3, text_features)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.prompt_guidance_2(inp_dec_level2, text_features)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.prompt_guidance_1(inp_dec_level1, text_features)
        # print(inp_dec_level1.shape)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


    @torch.no_grad()
    def get_image_feature(self, img):
        # print(img.size())
        image_features = self.model_clip.encode_image(img)
        return image_features


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    # model = MambaDFuse(upscale=2, img_size=(height, width),
    #                window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
    #                embed_dim=60, num_heads=[6, 6, 6, 6])
    model = SingleMambaBlock()

    x = torch.randn((1, 3, height, width))
    x = model(x)

