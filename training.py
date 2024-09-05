"""
This module defines
logic to train a given
model with the given training
params
"""

import numpy as np
import torch
import torch.nn as nn
import tyro
import monai
from monai.losses import DiceLoss

from utils import train_test_split, Lambda, printLambda

def verifyShapeSupport(modelFunc):
  """
  This function can check if a model
  supports the shape we expect it
  to support for batch inference.
  """
  batch_size = 2
  width = 240
  height = 240
  numslices = 155
  depth = 4

  randomInput = torch.randn( (batch_size, width, height, numslices, depth) )
  
  try:
    modelOut = modelFunc(randomInput)

    # Verify shape matches [b, w, h, numSlices, 1]
    assert modelOut.shape == (batch_size, width, height, numslices), f"Expected shape (b, w, h, numSlices), got {modelOut.shape}"
  except:
    raise ValueError(f"Model does not support the expected shape; expected (b, w, h, numSlices) but got {modelOut.shape}")


def train_model(modelFunc, model, BATCH_SIZE : int = 10, LR : float = 1e-3, EPOCHS : int = 20):
  """
  Our main training loop.

  :param modelFunc: A function that takes in a tensor
    of shape (B, W, H, numSlices, 4), and outputs a tensor
    of shape (B, W, H, numSlices, 1) that contains the segmentation
    prediction
  :param model: The model to train, as an nn.Module
  :param BATCH_SIZE: The batch size to use
  :param LR: The learning rate to use
  :param EPOCHS: The number of epochs to train for
  """
  
  verifyShapeSupport(modelFunc)

  # First, build up our data utils
  trainDataset, testDataset = train_test_split()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)

  trLoad = monai.data.DataLoader(trainDataset, batch_size=BATCH_SIZE)
  tsLoad = monai.data.DataLoader(testDataset, batch_size=BATCH_SIZE)

  # Get some examples we can use for reshaping
  EX_IN, EX_OUT = trainDataset[0]

  optim = torch.optim.Adam(model.parameters(), lr=LR)

  diceLoss = DiceLoss(reduction="mean", sigmoid=True)

  # Keep track of our train loss and
  # test loss, per epoch
  trainLosses = []
  testLosses = []

  # Now, train.
  for epochIdx in range(EPOCHS):
    currEpochTrainLosses = []
    for batchIdx, batch in enumerate(iter(trLoad)):
      inp, out = batch

      inp = inp.to(device, non_blocking=True)
      out = out.to(device, non_blocking=True)

      modelPred = modelFunc(inp)

      loss = diceLoss(modelPred, out)
      print(loss.item())
      currEpochTrainLosses.append(loss.item())

      optim.zero_grad()
      loss.backward()
      optim.step()

    trainLosses.append(np.mean(currEpochTrainLosses))

    # Now, run our test set, and get
    # average DICE loss for this epoch
    with torch.no_grad():
      currEpochTestLosses = []
      for batchIdx, batch in enumerate(iter(tsLoad)):
        inp, out = batch
        inp = inp.to(device, non_blocking=True)
        out = out.to(device, non_blocking=True)
        modelPred = modelFunc(inp)
        loss = diceLoss(modelPred, out)

        currEpochTestLosses.append(loss.item())

      
      testLosses.append(np.mean(currEpochTestLosses[0]))

  # Now, after all our epochs, plot
  # our train and test loss,
  # and then save to disk
  import matplotlib.pyplot as plt
  epochs = range(EPOCHS)
  plt.plot(epochs, trainLosses, label="Train Loss")
  plt.plot(epochs, testLosses, label="Test Loss")
  plt.savefig("loss.png")

  print(epochs, trainLosses, testLosses)

  return model
