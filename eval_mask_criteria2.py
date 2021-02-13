import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset_train import DAVIS_MO_Test
from models import resnet18
from models.selector_net import selector_net
import os
import itertools
import cv2

default_height = 256
default_width = 512

solve = selector_net()
solve.eval()
pth_path = "/home/vidit/models/UnVOS/checkpoint/soft_lab/epoch2/model.ckpt"
model_dict = torch.load(pth_path)

solve.model.load_state_dict(model_dict["resnet"])
solve.fc1.load_state_dict(model_dict["fc1"])
solve.fc2.load_state_dict(model_dict["fc2"])

def preprocess_image(rgb_img, curr_stm, curr_mask_rcnn):

  rgb_img = rgb_img/255.0
  mask_crop = cv2.resize(curr_mask_rcnn, (default_width, default_height), interpolation=cv2.INTER_NEAREST)
  stm_crop = cv2.resize(curr_stm, (default_width, default_height), interpolation=cv2.INTER_NEAREST)
  rgb_crop = cv2.resize(rgb_img, (default_width, default_height))

  mask_crop = np.reshape(mask_crop, (mask_crop.shape[0], mask_crop.shape[1], 1))
  stm_crop = np.reshape(stm_crop, (stm_crop.shape[0], stm_crop.shape[1], 1))
  
  mask_input = np.concatenate((rgb_crop, mask_crop), -1)
  stm_input = np.concatenate((rgb_crop, stm_crop), -1)

  input_dict = {}
  input_dict["input1"] = (torch.from_numpy(np.transpose(mask_input, (2, 0, 1))).float()).view(1, 4, rgb_crop.shape[0], rgb_crop.shape[1])
  input_dict["input2"] = (torch.from_numpy(np.transpose(stm_input, (2, 0, 1))).float()).view(1, 4, rgb_crop.shape[0], rgb_crop.shape[1])
  return input_dict 


def process_single_image(rgb_img, stm_mask, mask_rcnn_mask):

  data = preprocess_image(rgb_img, stm_mask, mask_rcnn_mask)
  img1, img2 = data["input1"], data["input2"] 
  img1, img2 = img1.to("cuda:0"), img2.to("cuda:0")  

  with torch.no_grad():
    output = solve(img1, img2)

  return output.cpu().detach().numpy()[0]
