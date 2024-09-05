import torch
import torch.nn as nn
import tyro
import monai
from monai.losses import DiceLoss

from utils import train_test_split, Lambda, printLambda
from training import train_model

def train_linear(BATCH_SIZE : int = 10, LR : float = 1e-3, EPOCHS : int = 20, DOWNSCALE_FACTOR : int =4):

  # Generate an example input
  # and output tensor, just
  # so we have an example
  EX_IN = torch.zeros(240, 240, 155, 4)
  EX_OUT = EX_IN
  
  # Input shape is (B, H, W, SLICES, 4), but after vmapping it's just (H, W, 4)
  linear = nn.Sequential(
    Lambda(lambda x : x.unsqueeze(0)), # Unsqueeze once to turn into 1, H, W, 4
    Lambda(lambda x : torch.permute(x, (0, 3, 1, 2))), # Now it's 1, 4, H, W
    nn.AvgPool2d(DOWNSCALE_FACTOR), # Use average pool to downsample to 1, 4, H // DOWNSCALE_FACTOR, W // DOWNSCALE_FACTOR
    nn.Flatten(), # Going to be size 4 * H // DOWNSCALE_FACTOR * W // DOWNSCALE_FACTOR
    nn.Linear(4 * EX_IN.shape[0] // DOWNSCALE_FACTOR * EX_IN.shape[1] // DOWNSCALE_FACTOR, EX_IN.shape[0] // DOWNSCALE_FACTOR * EX_IN.shape[1] // DOWNSCALE_FACTOR), # A linear layer
    Lambda(lambda x: x.reshape(1, 1, EX_IN.shape[0] // DOWNSCALE_FACTOR, EX_IN.shape[1] // DOWNSCALE_FACTOR)), # Reshape to 1, H, W
    nn.Upsample((EX_OUT.shape[0], EX_OUT.shape[1])), # Upscale to original size
    Lambda(lambda x : x.reshape(EX_OUT.shape[0], EX_OUT.shape[1])), # Lose the missing dimension
    )

  vm = torch.vmap

  # To make model work on our shape(B, H, W, slices, 4 channels), we map over the slices and batch, then
  # do all our other work going from all 4 input dimensions to 1 output dimension
  _vmodel = vm(vm(linear, in_dims=(-2)))
  model = lambda x : torch.permute(_vmodel(x), (0, 2, 3, 1))

  # Now, train.
  train_model(model, linear, BATCH_SIZE, LR, EPOCHS)

train_linear()

#if __name__ == "__main__":
  #tyro.cli(train_linear)
