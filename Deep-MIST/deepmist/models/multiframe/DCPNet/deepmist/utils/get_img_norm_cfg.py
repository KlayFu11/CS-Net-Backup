#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Rui Gao
# @time    : 2023/11/8 23:45

import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def list_all_img_paths(data_root, img_foramt):
    img_paths = []
    # for seq in os.listdir(data_root):
    #     img_paths += glob(os.path.join(data_root, seq, img_foramt))
    for seq in [27,59,12,21,35,91,71,18,30,86,43,70,22,11,17,48,69,50,82,73,65,76,92,77,87,57,2,80,55,72,75,64,32,7,10,85,19,16,31,23,4,13,58,81,68,88,84,56,67,28,9,3,83,79,44,49,52,74]:
        img_paths += glob(os.path.join(data_root, str(seq), img_foramt))
        pass
    return img_paths


# def get_img_norm_cfg(dataset_root, img_format):
#     img_path_list = []
#     trainset_root = os.path.join(dataset_root, 'train')
#     testset_root = os.path.join(dataset_root, 'test')
#     img_path_list += list_all_img_paths(trainset_root, img_format)
#     img_path_list += list_all_img_paths(testset_root, img_format)
#
#     mean_list = []
#     std_list = []
#     for img_pth in img_path_list:
#         img = Image.open(img_pth).convert('I')
#         img = np.array(img, dtype=np.float32) / 255
#         mean_list.append(img.mean())
#         std_list.append(img.std())
#
#     img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
#     return img_norm_cfg


def get_img_norm_cfg(root, img_format):
    img_path_list = []
    seqs_dir = os.path.join(root, 'images')
    img_path_list += list_all_img_paths(seqs_dir, img_format)

    img_list = []

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()])
    for img_path in img_path_list:
        img = Image.open(img_path).convert('RGB')  # 'RGB', 'I' or 'L'
        img_list.append(transform(img))

    stacked_imgs = torch.stack(img_list)
    mean = []
    std = []
    for c in range(stacked_imgs.shape[1]):
        mean.append(round(stacked_imgs[:, c, :, :].mean().item(), 3))
        std.append(round(stacked_imgs[:, c, :, :].std().item(), 3))
    return dict(mean=mean, std=std)


if __name__ == '__main__':
    root = '/media/kou/Data2/zmh/dataset/IRDST'
    img_format = '*.png'

    img_norm_cfg = get_img_norm_cfg(root, img_format)
    print(img_norm_cfg)
