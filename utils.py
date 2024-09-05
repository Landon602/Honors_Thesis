"""
Some general utilities
to make working with
this data a breeze.
"""
import os
import random
import math

import torch
import torch.nn as nn
import numpy as np
import monai
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.data import Dataset, DataLoader

IMGS = "/Users/landonmattingly/Desktop/Honors_Thesis/imagesTr"
LABELS = "/Users/landonmattingly/Desktop/Honors_Thesis/labelsTr"

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def printF(X):
  print(X.shape)
  return X

def printVal(X):
  print(X)
  return X

def printDist(X):
  """
  Prints minimum, maximum,
  and other stuff about X
  """
  print(f"Min: {X.min()}")
  print(f"Max: {X.max()}")
  print(f"Mean: {X.mean()}")
  print(f"Std: {X.std()}")
  return X

printLambda = Lambda(printF)
distLambda = Lambda(printDist)
valLambda = Lambda(printVal)

img_transforms = monai.transforms.Compose([
  ScaleIntensity(),
  ToTensor()
])

mask_transforms = monai.transforms.Compose([
  ToTensor(),
  lambda x: x != 0,
])

def train_test_split(trainFrac : float = 0.8):
  """
  Given a train-test split fraction,
  returns 2 datasets, the first being
  for the train and test dataset
  """
  # First, get all names in imagesTR
  fNames = set(os.listdir(IMGS))

  # Restrict to nii only
  fNames = [name for name in fNames if "nii" in name]

  # Now, pick train and test fnames
  trainFNames = random.sample(fNames, k=math.ceil(len(fNames) * trainFrac))
  testFNames = set(fNames) - set(trainFNames)

  trainFNames_ = [f"{IMGS}/{name}" for name in trainFNames]
  trainSegNames_ = [f"{LABELS}/{name}" for name in trainFNames]
  testFNames_ = [f"{IMGS}/{name}" for name in testFNames]
  testSegNames_ = [f"{LABELS}/{name}" for name in testFNames]
  
  # Create a train and test dataset
  # In monai
  trainSet = monai.data.ImageDataset(trainFNames_, seg_files=trainSegNames_, transform=img_transforms, seg_transform=mask_transforms,)
  testSet = monai.data.ImageDataset(testFNames_, seg_files=testSegNames_, transform=img_transforms, seg_transform=mask_transforms,)

  return trainSet, testSet

if __name__ == "__main__":
  trS, tsS = train_test_split()
  
  trLoad = monai.data.DataLoader(trS, batch_size=10, )
  tsLoad = monai.data.DataLoader(tsS, batch_size=10, )
  
  trImg, trMask = trS[0]
  
  print(trMask.shape)
  
  import matplotlib.pyplot as plt
  
  for i in range(1):
    plt.subplot(1, 4, i + 1)
    plt.imshow(trMask[:, :, 50].float())
    plt.title(f"Mask channel {i}")

  plt.show()
  
  for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(trImg[:, :, 50, i].float(), cmap="gray")
    plt.title(f"Image channel {i}")
  
  plt.show()
