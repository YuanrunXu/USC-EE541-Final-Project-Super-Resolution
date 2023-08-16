import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, hr_folder, lr_folder):

        # hr_list = os.listdir(hr_folder)
        # print(hr_list)
        # hr_list = hr_list.sort()
        # print(hr_list)
        hr_list = os.listdir(hr_folder)
        hr_list.sort()

        lr_list = os.listdir(lr_folder)
        lr_list.sort()

        # print(hr_list)
        # print(lr_list)
        self.hr_image_paths = [os.path.join(hr_folder, img_name) for img_name in hr_list]
        # print(self.hr_image_paths)
        # self.hr_image_paths = [os.path.join(hr_folder, img_name) for img_name in hr_list]

        # lr_list = os.listdir(hr_folder)
        # print(lr_list)
        # lr_list = lr_list.sort()
        self.lr_image_paths = [os.path.join(lr_folder, img_name) for img_name in lr_list]
        # print(self.lr_image_paths)

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, index):
        # print(index)
        hr_image_path = self.hr_image_paths[index]
        # print(index)
        lr_image_path = self.lr_image_paths[index]

        # #index is the same, but the images get are not pair
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_COLOR)
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_COLOR)
        #
        # from matplotlib import pyplot as plt
        # print(hr_image_path)
        # plt.imshow(hr_image)
        # plt.show()
        # print('lr')
        # print(lr_image_path)
        # plt.imshow(lr_image)
        # plt.show()

        # Grad
        # hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY)

        # Color
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Resize
        hr_target_size = (224, 224)
        lr_target_size = (112, 112)
        hr_image = cv2.resize(hr_image, hr_target_size, interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, lr_target_size, interpolation=cv2.INTER_CUBIC)

        # Normalization
        hr_image = torch.tensor(hr_image, dtype=torch.float32) / 255.0
        lr_image = torch.tensor(lr_image, dtype=torch.float32) / 255.0

        hr_image = hr_image.permute(2, 0, 1)
        lr_image = lr_image.permute(2, 0, 1)

        return hr_image, lr_image


train_hr_folder = "../data/DIV2K/DIV2K_train_HR"
train_lr_folder = "../data/DIV2K/DIV2K_train_LR_bicubic/X2"

valid_hr_folder = "../data/DIV2K/DIV2K_valid_HR"
valid_lr_folder = "../data/DIV2K/DIV2K_valid_LR_bicubic/X2"

test_hr_folder = "../data/DIV2K/DIV2K_test_HR"
test_lr_folder = "../data/DIV2K/DIV2K_test_LR_bicubic/X2"


train_dataset = SRDataset(train_hr_folder, train_lr_folder)
valid_dataset = SRDataset(valid_hr_folder, valid_lr_folder)
test_dataset = SRDataset(test_hr_folder, test_lr_folder)