import torch
import torch.nn as nn
import torchvision
import tyro
import monai
from monai.losses import DiceLoss

from utils import train_test_split, Lambda, printLambda
from training import train_model

def train(BATCH_SIZE : int = 10, LR : float = 1e-3, EPOCHS : int = 20, DOWNSCALE_FACTOR : int =4):
    myModel = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    for param in myModel.parameters():
        param.requires_grad = False

    num_ftrs = myModel.classifier.in_features
    myModel.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs,2)
        )
    def modelFunc(X):
        X = torch.permute(X, (0, 4, 1, 2, 3))
        Y = torch.squeeze(myModel(X), dim=2)
        Y = torch.permute(Y, (1, 2, 3, 0))
        return Y
    
    train_model(modelFunc, myModel, BATCH_SIZE, LR, EPOCHS)

print("test")