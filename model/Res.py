import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from Transformer import TransformerBlock


class resblock(nn.Module):  ## that is a part of model
    def __init__(self, inchannel, outchannel, stride=1):
        super(resblock, self).__init__()
        ## conv branch
        self.left = nn.Sequential(  ## define a serial of  operation
            nn.Conv2d(inchannel, outchannel, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(outchannel))
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel))

    ### get the residual
    def forward(self, x):
        return F.relu(self.left(x) + self.short_cut(x))



# class Dexblock(nn.Module):
#     def __init__(self, in_channel,out_channel,num):
#         super(Dexblock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel,out_channel,3,1,1)
#         self.TransformerBlock = nn.Sequential(*[TransformerBlock(dim=int(out_channel), num_heads=8, ffn_expansion_factor=2.66,
#                              bias=False, LayerNorm_type='WithBias') for i in range(num)])
#         self.MambaBlock = nn.Sequential(*[SingleMambaBlock(int(out_channel)) for i in range(num)])
#         self.DenseBlock = nn.Sequential(*[resblock(out_channel,out_channel) for i in range(num)])
#         self.dconv = nn.Conv2d(out_channel*2,out_channel,3,2,1)
#     def forward(self,x):
#         acctivate = nn.LeakyReLU()
#         out = acctivate(self.conv1(x))
#         out = self.TransformerBlock(out)
#         outglobal = self.MambaBlock(out)
#         outlocal = self.DenseBlock(out)
#         out = self.dconv(torch.cat([outglobal,outlocal],1))
#         return out

if __name__ == '__main__':
    img = torch.randn((1, 3, 128, 128)).cuda()
    model = Dexblock(3,16).cuda()
    out = model(img)
    print(out.shape)
