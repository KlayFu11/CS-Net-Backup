import os
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
from deepmist.utils import (rgb_loader, binary_loader, random_flip, random_crop, random_rotation, color_enhance,
                            random_peper)

class IRDSTDataset(Data.Dataset):
    def __init__(self, root, num_inputs=5, img_size=None, frame_padding=False, data_aug=None, mode='train'):
        if mode == 'train':
            seq_list = [27,59,12,21,35,91,71,18,30,86,43,70,22,11,17,48,69,50,82,73,65,76,92,
                        77,87,57,2,80,55,72,75,64,32,7,10,85,19,16,31,23,4]

        elif mode == 'val_all':
            seq_list = [13,58,81,68,88,84,56,67,28,9,3,83,79,44,49,52,74]
        # if mode == 'train':
        #     seq_list = [
        #         1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        #         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        #         21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        #         31, 32, 33, 34, 35, 36, 38, 40,
        #         41, 42, 43, 44, 45, 46, 48, 49, 50,
        #         51, 52, 55, 56, 57, 58, 59,
        #         61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        #         71, 72, 73, 74
        #     ]

        # elif mode == 'val_all':
        #     seq_list = [
        #         75, 76, 77, 79, 80, 81, 82, 83, 84,
        #         85, 86, 87, 88, 89, 90, 91, 92
        #     ]

        elif mode == 'val_lSCR':
            seq_list = []

        elif mode == 'val_hSCR':
            seq_list = []
        else:
            raise ValueError(f"Invalid mode '{mode}'. It must be 'train', 'val_all', 'val_lSCR' or 'val_hSCR'.")

        if img_size is None:
            img_size = [400, 400]
        self.img_size = img_size

        if data_aug is None:
            data_aug = {
                'random_flip': True,
                'random_crop': True,
                'random_rotation': True,
                'color_enhance': False,
                'random_peper': False
            }
        self.num_inputs = num_inputs
        self.data_aug = data_aug
        self.mode = mode
        self.grouped_frame_paths = []
        self.grouped_mask_paths = []

        # grouping
        for seq in seq_list:
            frame_paths = sorted(glob(os.path.join(root, 'images', str(seq), '*.png')))
            mask_paths = sorted(glob(os.path.join(root, 'masks', str(seq), '*.png')))
            num_frames = len(frame_paths)
            assert num_inputs <= num_frames, f"number of input frames '{num_inputs}' exceeds the total number."
            if frame_padding:
                for i in range(num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        if i - j < 0:
                            frame_list.append(frame_paths[0])
                        else:
                            frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])
            else:
                for i in range(num_inputs - 1, num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])

        # transforms
        self.frame_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.359, 0.359, 0.359],[0.15, 0.15, 0.15])])  # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                                        # MIST:[0.359, 0.359, 0.359],[0.15, 0.15, 0.15]
                                                        # IRDST:{'mean': [0.368, 0.368, 0.368], 'std': [0.153, 0.153, 0.153]}
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])

    def __getitem__(self, index):
        frames = []
        for i in range(self.num_inputs):
            frames.append(rgb_loader(self.grouped_frame_paths[index][i]))
        mask = binary_loader(self.grouped_mask_paths[index])

        # data augmentation (only for training)
        if self.mode == 'train':
            if self.data_aug['random_flip']:
                frames, mask = random_flip(frames, mask)
            if self.data_aug['random_crop']:
                frames, mask = random_crop(frames, mask)
            if self.data_aug['random_rotation']:
                frames, mask = random_rotation(frames, mask)
            if self.data_aug['color_enhance']:
                frames = color_enhance(frames)
            if self.data_aug['random_peper']:
                mask = random_peper(mask)

        # transforms
        for i in range(len(frames)):
            frames[i] = self.frame_transform(frames[i])
        frames = torch.stack(frames, dim=1)
        mask = self.mask_transform(mask)

        # get name
        mask_path_split = self.grouped_mask_paths[index].split('/')
        name = mask_path_split[-2] + '/' + mask_path_split[-1]

        # return frames, mask, name
        return frames, mask, self.img_size[0], self.img_size[1], name

    def __len__(self):
        return len(self.grouped_frame_paths)
