import numpy as np
import pandas as pd
import shutil
import os
import random
import zipfile
import torch
import torch.nn as nn
import csv
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch.nn.functional as F
import copy
import tqdm
import time
from PIL import Image

from seedEverything import SeedEverything

import albumentations
from albumentations import pytorch as AT


def infer(modelDataPath, datasetPath):
    SeedEverything(41)

    labelList = ['baseball', 'formula1', 'fencing', 'motogp', 'ice_hockey',# был ранее получен функцией GetCategoryList
                 'wrestling', 'boxing', 'volleyball', 'cricket', 'basketball', 'wwe',
                 'swimming', 'weight_lifting', 'gymnastics', 'tennis', 'kabaddi', 'badminton',
                 'football', 'table_tennis', 'hockey', 'shooting', 'chess']

    test_files = os.listdir(datasetPath)
    print("Test set size: ", len(test_files))  # 1645
    os.getcwd()
    class SportsDataset(Dataset):
        def __init__(self, file_list, dir, transform=None):
            self.file_list = file_list
            self.dir = dir
            self.transform = transform

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            image = cv2.imread(os.path.join(self.dir, self.file_list[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            return image

    img_size = 256

    data_transforms_test = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    test_set = SportsDataset(test_files, datasetPath, data_transforms_test)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             num_workers=0, shuffle=False)

    samples = next(iter(testloader))
    plt.figure(figsize=(16, 24))
    grid_imgs = torchvision.utils.make_grid(samples[:24])
    np_grid_imgs = grid_imgs.numpy()
    plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    model = torchvision.models.resnet152(pretrained=True, progress=True)
    model.fc = nn.Linear(2048, 1024)
    model.fc1 = nn.Linear(1024, 512)
    model.fc2 = nn.Linear(512, 22)
    model.load_state_dict(torch.load(modelDataPath))
    model.eval()

    model = model.to(device)

    print("Classification started")

    model.eval()
    f = open("output.csv", "w")
    with torch.no_grad():
        for i, image in enumerate(testloader, 0):
            image = image.to(device=device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            sample_fname = testloader.dataset.file_list[i]
            line = datasetPath + "\\" + sample_fname + "," + str(labelList[predicted.item()]) + '\n'
            f.write(line)
    f.close()
