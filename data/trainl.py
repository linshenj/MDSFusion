import os
from random import random

import torchvision.transforms as transforms
import torch
from PIL import Image, ImageEnhance, ImageChops
from PIL.ImageDraw import ImageDraw
from torch.utils import data
from torchvision import transforms
import numpy as np

to_tensor = transforms.Compose([transforms.ToTensor()])
to_pil_image = transforms.ToPILImage()

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr
class msrs_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'vi':
                self.vis_path = temp_path  # 获得可见光路径
            elif sub_dir =='vis_en':
                self.enhanced_path = temp_path  # 获得增强路径
            elif sub_dir == 'ir':
                self.inf_path = temp_path  #获得红外图像路径



        self.name_list = os.listdir(self.vis_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):

        name = self.name_list[index]  # 获得当前图片的名称
        vis_image = Image.open(os.path.join(self.vis_path, name))  # 获取可见光图像
        enhanced_image = Image.open(os.path.join(self.enhanced_path, name))  #获取增强
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        vis_image = self.transform(vis_image)
        enhanced_image = self.transform(enhanced_image)
        inf_image = self.transform(inf_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)

        enhanced_y_image, _, _ = RGB2YCrCb(enhanced_image)


        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, enhanced_y_image, inf_image
    def __len__(self):
        return len(self.name_list)



