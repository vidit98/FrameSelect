from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
from scipy.optimize import linear_sum_assignment
import sys
import random
 
### My libs
from dataset import DAVIS_MO_Test
from model.stm import STM

torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    parser.add_argument("-v", type=str, help="name of video",default='')
    return parser.parse_args()

args = get_arguments()

GPU = args.g
SET = args.s
DATA_ROOT = args.D
MASK_ROOT= os.path.join(DATA_ROOT, 'Annotations/480p')
vid_name = args.v

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on DAVIS')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')


def remove_extra(pred):

    #this funtion removes small extra noisy propogations from masks of every object

    pred[0,:] = 0
    pred[:, 0] = 0
    pred[-1, :] = 0
    pred[:, -1] = 0
    unique_l = np.unique(pred[pred!=0])
    new_pred = np.zeros_like(pred)

    for l in unique_l:
        curr_k = np.uint8(pred == l)
        contours,_ = cv2.findContours(curr_k, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        all_area = [cv2.contourArea(c) for c in contours]
        max_area = max(all_area)
        
        #remove smal noisy propogation for each object
        final_contour = [contours[i] for i in range(len(contours)) if all_area[i] > 0.25*max_area]
        cv2.drawContours(new_pred, final_contour, -1, int(l), -1)

    return new_pred

def calc_intersect(curr_pred, mask_curr_pred):

    unique_l = np.unique(curr_pred[curr_pred!=0])
    unique_m = np.unique(mask_curr_pred[mask_curr_pred!=0])

    cost = np.zeros((len(unique_l), len(unique_m)))

    for i in range(len(unique_l)):
        for j in range(len(unique_m)):
            num = float(np.sum(np.logical_and(mask_curr_pred==unique_m[j], curr_pred==unique_l[i])))
            denom = float(np.sum(np.logical_or(mask_curr_pred==unique_m[j], curr_pred==unique_l[i])))
            cost[i, j] = 1 - num/denom

    row_ind, col_ind = linear_sum_assignment(cost)
    final_r = []
    final_c = []

    for i in range(len(row_ind)):
        if cost[row_ind[i], col_ind[i]] <= 0.5:
            final_r.append(unique_l[row_ind[i]])
            final_c.append(unique_m[col_ind[i]])

    return final_r, final_c

def update_pred(Es, t, seq_name, no_obj, tot_obj):

    #updates current prediction using criteria 2

    curr_pred = np.argmax(Es[0, :, t], axis=0).astype(np.uint8)

    curr_pred = remove_extra(curr_pred)
    new_curr_pred = np.copy(curr_pred)  # used to store the new prediction 
    prev_pred = np.argmax(Es[0, :, t-1], axis=0).astype(np.uint8)

    #read current mask rcnn precition
    mask_curr_pred = cv2.imread(os.path.join(MASK_ROOT, seq_name, "%05d"%(t) + ".png"), 0)

    #hungarian maching b/w stm and mask rcnn masks
    assoc_r, assoc_c = calc_intersect(curr_pred, mask_curr_pred)
    unique_m = np.unique(mask_curr_pred[mask_curr_pred!=0])
    unique_p = np.unique(curr_pred[curr_pred!=0])

    used = []
    for l in unique_p:
        prev_area = np.sum(prev_pred == l)
        if prev_area > 0:
            curr_area = np.sum(curr_pred == l)

            if l in assoc_r:
                l_ind = assoc_r.index(l)
                idw = assoc_c[l_ind]
                used += [idw]
                m_area = np.sum(mask_curr_pred == idw)
                change_area_l = np.fabs(curr_area - prev_area)/prev_area
                change_area_m = np.fabs(m_area - prev_area)/prev_area

                if change_area_m < change_area_l:
                    new_curr_pred[curr_pred == l] = 0
                    f_mask = np.logical_and(new_curr_pred==0, mask_curr_pred==idw)
                    new_curr_pred[f_mask] = l


    for m in unique_m:
    	if m in used:
    		continue

        #checking the new object masks
           
    	obj = np.sum(np.logical_and(mask_curr_pred == m, new_curr_pred!=0))
    	m_area = np.sum(mask_curr_pred == m)
    	
    	thresh = 0.1  #threshold for maximum percentage of intersection with the already added objects

    	if obj < thresh*m_area and no_obj < tot_obj:
            m_color = no_obj[0] + 1
            unique_p =np.append(unique_p, m_color)
            new_curr_pred[mask_curr_pred == m] = m_color
            no_obj = no_obj + 1

    new_Es_t = np.zeros((Es.shape[0], Es.shape[1], Es.shape[3], Es.shape[4]), dtype=np.float32)
    unique_l_b = np.unique(new_curr_pred)
    
    for i in unique_l_b:
        new_Es_t[0, i] = (new_curr_pred == i).astype(np.float32)

    Es[:, :, t] = new_Es_t
    
    return torch.from_numpy(Es).float(), no_obj


def Run_video(Fs, Ms, num_frames, num_objects, seq_name, Mem_every=None, Mem_number=None):

    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    #initial number of objects
    num_objects_init = num_objects

    #max_number_of_objects after addition
    num_objects = num_objects + 3


    for t in tqdm.tqdm(range(1, num_frames)):

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects])) 

        if t-1 == 0: #
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)

        # segment
        with torch.no_grad():
            logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)

        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values

        Es, num_objects_init = update_pred(Es.cpu().numpy(), t, seq_name, num_objects_init, num_objects)

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    
    return pred, Es



Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', vid_name=vid_name)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = '{}_DAVIS_{}{}'.format(MODEL,'2019',SET)
print('Start Testing:', code_name)


for seq, V in enumerate(Testloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
    
    pred, Es = Run_video(Fs, Ms, num_frames, num_objects, seq_name, Mem_every=5, Mem_number=None)
        
    # Save results ######################
    test_path = os.path.join('./results', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))
