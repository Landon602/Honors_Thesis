import torch
import torch.nn as nn
import tyro
import monai
from monai.losses import DiceLoss

from utils import train_test_split, Lambda, printLambda
from training import train_model

def train(BATCH_SIZE : int = 10, LR : float = 1e-3, EPOCHS : int = 20, DOWNSCALE_FACTOR : int =4):
  # For CNN, the initial input is
  # of shape [B, H, W, numSlices, 4].
  #BatchSize, Height, Width, numSlices (155), channels (4)
  # Then, apply vmap on slices, and permute
  #Don't think too hard about Vmap, just run it
  # to shape [B, 4, H, W]

  # So, in this model, assume we have input
  # shape of [B, 4, H, W], and do CNN
  # magic
  cnn = nn.Sequential(
    nn.Conv2d(4, 8, 4),
    nn.ReLU(),
    nn.Conv2d(8, 10, 3),
    nn.ReLU(),
    nn.ConvTranspose2d(10, 8, 3),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4),
    nn.ReLU(),
    nn.Conv2d(4, 1, 1)
  )

  _vmappedCnn = torch.vmap(cnn, in_dims=-1)

  def modelFunc(X):
    print(X.shape)
    X = torch.permute(X, (0, 4, 1, 2, 3))
    print(X.shape)
    Y = torch.squeeze(_vmappedCnn(X), dim=2)
    print(Y.shape)
    Y = torch.permute(Y, (1, 2, 3, 0))
    return Y
    
  train_model(modelFunc, cnn, BATCH_SIZE, LR, EPOCHS)


if __name__ == "__main__":
  tyro.cli(train)
