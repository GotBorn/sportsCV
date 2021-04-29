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


def infer(modelFolderDataPath, datasetPath):
    SeedEverything(41)

    labelList = ['baseball', 'formula1', 'fencing', 'motogp', 'ice_hockey',
                 # был ранее получен функцией GetCategoryList
                 'wrestling', 'boxing', 'volleyball', 'cricket', 'basketball', 'wwe',
                 'swimming', 'weight_lifting', 'gymnastics', 'tennis', 'kabaddi', 'badminton',
                 'football', 'table_tennis', 'hockey', 'shooting', 'chess']

    test_files = os.listdir(datasetPath)

    # сортировка файлов в правильном порядке
    for i in range(0, len(test_files)):
        test_files[i] = int(test_files[i].replace(".jpg", ""))

    test_files.sort()

    for i in range(0, len(test_files)):
        test_files[i] = str(test_files[i]) + ".jpg"

    print("Test set size: ", len(test_files))  # 1645

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    ##########
    modelResnet = torchvision.models.resnet152(pretrained=True, progress=True)
    modelResnet.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 22)
    )
    ###########
    modelResnext = torchvision.models.resnext50_32x4d(pretrained=True, progress=True)
    modelResnext.fc = nn.Sequential(
        nn.Linear(2048, 1500),
        nn.LeakyReLU(),
        nn.Linear(1500, 800),
        nn.LeakyReLU(),
        nn.Linear(800, 300),
        nn.LeakyReLU(),
        nn.Linear(300, 22)
    )
    ###########
    modelGoogleNet = torchvision.models.googlenet(pretrained=True, progress=True)
    modelGoogleNet.fc = nn.Sequential(
        nn.Linear(1024, 800),
        nn.LeakyReLU(),
        nn.Linear(800, 500),
        nn.LeakyReLU(),
        nn.Linear(500, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 22)
    )
    ###########
    modelDenseNet = torchvision.models.densenet201(pretrained=True, progress=True)
    modelDenseNet.classifier = nn.Sequential(
        nn.Linear(1920, 1500),
        nn.LeakyReLU(),
        nn.Linear(1500, 1000),
        nn.LeakyReLU(),
        nn.Linear(1000, 400),
        nn.LeakyReLU(),
        nn.Linear(400, 22)
    )

    modelResnet.load_state_dict(torch.load(modelFolderDataPath + "/resnet152Model.pt"))
    modelResnext.load_state_dict(torch.load(modelFolderDataPath + "/ResnextModel.pt"))
    modelGoogleNet.load_state_dict(torch.load(modelFolderDataPath + "/GoogleNetModel.pt"))
    modelDenseNet.load_state_dict(torch.load(modelFolderDataPath + "/DenseNetModel.pt"))

    modelResnet.eval()
    modelResnext.eval()
    modelGoogleNet.eval()
    modelDenseNet.eval()

    modelResnet = modelResnet.to(device)
    modelResnext = modelResnext.to(device)
    modelGoogleNet = modelGoogleNet.to(device)
    modelDenseNet = modelDenseNet.to(device)

    print("Classification started")

    f = open("output.csv", "w")
    with torch.no_grad():
        for i, image in enumerate(testloader, 0):
            image = image.to(device=device)

            outputResnet = modelResnet(image)
            outputResnext = modelResnext(image)
            outputGoogleNet = modelGoogleNet(image)
            outputDenseNet = modelDenseNet(image)

            _, predicted = torch.max((outputResnet.data + outputResnext.data +
                                      outputGoogleNet.data + outputDenseNet.data) / 4, 1)
            sample_fname = testloader.dataset.file_list[i]
            line = sample_fname + "," + str(labelList[predicted.item()]) + '\n'
            f.write(line)
    f.close()
    print("Classification finished")
