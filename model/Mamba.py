import torch.nn as nn
from timm.models.layers import to_2tuple
from einops import rearrange
import numbers
from mamba_simple import Mamba
import torch


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

