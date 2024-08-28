"""测试融合网络"""
import argparse
import os
import random
import statistics
import time
import clip
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.testl import msrs_data


from model.CMRFusion import MKFusion
# from scripts.utils import clip_norm
# from train_seg import clamp

def clip_norm(im):
    DEV = im.device
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im_re = F.interpolate(im.repeat(1, 3, 1, 1) if im.shape[1] == 1 else im, size=224, mode='bilinear',
                          align_corners=False)
    im_norm = (im_re - mean) / std
    return im_norm
def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)
def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out
def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MT-fuse')
    parser.add_argument('--dataset_path', metavar='DIR', default='/mnt/f/A/Source-image/S',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('--save_path', default='output')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # parser.add_argument('--Net', default='./pretrain/net/the best.pth')
    parser.add_argument('--Net', default='./result/net/4.pth') #4
    # parser.add_argument('--decoder', default='./result/decoder/1.pth')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, process = clip.load('ViT-B/32', device)
    # model_clip.load_state_dict(torch.load("fine_tuned_clip.pth"))
    test_dataset = msrs_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    #######加载模型
    model_clip.eval()
    model = MKFusion()
    model = model.cuda()
    model.load_state_dict(torch.load(args.Net))
    model.eval
    fuse_time = []
    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for vis_image, vis_y_image, cb, cr,inf_image,name,c in test_tqdm:
            vis_y_image = vis_y_image.cuda()
            inf_image = inf_image.cuda()
            vis_image = vis_image.cuda()
            # inf_image = torch.cat([inf_image]*3,1)
            cb = cb.cuda()
            cr = cr.cuda()
            # print(vis_image.shape)
            start = time.time()
            _,_,H, W = c.shape
            #########编码
            fused = model(vis_y_image, inf_image)
            end = time.time()
            fuse_time.append(end - start)
            ###########转为rgb
            fused = clamp(fused)
            # rgb_fused_image = YCrCb2RGB(fused[0], cb[0], cr[0])
            rgb_fused_image = YCrCb2RGB(fused[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image.squeeze(0))
            rgb_fused_image = rgb_fused_image.resize((W,H))
            rgb_fused_image.save(f'{args.save_path}/{name[0]}')
    mean = statistics.mean(fuse_time[1:])
    print(f'fuse avg time: {mean:.4f}')



