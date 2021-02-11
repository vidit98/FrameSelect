import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision
from tqdm import tqdm
from torchvision import transforms

class selector_net(nn.Module):
  def __init__(self):
    super(selector_net, self).__init__()
    self.model = resnet18().to("cuda:0")
    self.fc1 = nn.Linear(1024, 512).to("cuda:0")
    self.dropout1 = nn.Dropout(p=0.2).to("cuda:0")
    self.fc2 = nn.Linear(512,2).to("cuda:0")
    self.softm = nn.Softmax(dim=1).to("cuda:0")

  def forward(self, img1, img2):
    feat1 = self.model(img1)
    feat2 = self.model(img2)
    feat = torch.cat((feat1, feat2), 1) 
    feat = self.fc1(feat)
    output = self.fc2(self.dropout1(feat))
    output = self.softm(output)

    return output
