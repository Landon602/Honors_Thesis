import torch
import torch.nn as nn
import tyro
import monai
from monai.losses import DiceLoss

from utils import train_test_split, Lambda, printLambda
from training import train_model

def train(BATCH_SIZE : int = 10, LR : float = 1e-3, EPOCHS : int = 20, DOWNSCALE_FACTOR : int = 4):
  # As with the rest of the models,
  # we get an input shape of [B, H, W, numSlices, 4].
  # We are supposed to give an output of shape [B, H, W, numSlices]

  # For this one, I'll use a Unet off of Monai,
  # which has these implemented for us. In monai,
  # they expect a shape of [Batch size, input channels, w, h],
  # so I can use just one vmap and permute
  unet = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=4,
    out_channels=1,
    channels=(8, 16, 32),
    strides=(2, 2),
    num_res_units=2
  )

  vmappedUnet = torch.vmap(unet, in_dims=-1)

  def modelFunc(X):
    # First, reshape the input
    # to shape [B, 4, H, W, numSlices]
    print(X.shape)
    X = torch.permute( X, (0, 3, 1, 2, 4) )
    print(X.shape)
    exit(0)
    X = vmappedUnet(X)
    return X.squeeze(-1)
    
  train_model(modelFunc, unet, BATCH_SIZE, LR, EPOCHS)

if __name__ == "__main__":
  tyro.cli(train)
