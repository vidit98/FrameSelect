import os
import os.path as osp
import numpy as np
import cv2
import random

import torch
import torchvision
from torch.utils import data

import glob

default_height = 256
default_width = 512

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root="./video_seg/datasets/DAVIS", imset='train.txt', stm_dir='STM/train', mask_rcnn_dir='final_mrcnn', resolution='480p', single_object=False):
        self.root = root
        self.gt_dir = os.path.join(root, 'Annotations_final', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        self.stm_dir = stm_dir
        self.mask_rcnn_dir = mask_rcnn_dir
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, "2017", imset)

        self.map_video = {}
        i = 0

        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                imgs_paths = sorted(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                for path_no in range(len(imgs_paths)):
                    if path_no == 0:
                        continue
                    basename = os.path.basename(imgs_paths[path_no])
                    input_img = cv2.imread(imgs_paths[path_no], 1) 
                    gt_img = cv2.imread(os.path.join(self.gt_dir, _video, basename[:-3]+"png"), 0)
                    num_objects = np.max(gt_img)
                    for obj in range(1, num_objects+1):
                        curr_dict = {}
                        curr_dict['id'] = obj
                        curr_dict['video'] = _video
                        curr_dict['frame_no'] = int(basename[:-4])
                        self.map_video[i] = curr_dict
                        i = i + 1
                
                

    def __len__(self):
        return len(self.map_video)

    def __getitem__(self, index):
        curr_dict = self.map_video[index]
        video = curr_dict['video']
        frame_no = curr_dict['frame_no']
        object_no = curr_dict['id']
        rgb_img = cv2.imread(os.path.join(self.image_dir, video, "%05d"%(frame_no) + ".jpg"), 1)/255.0
        mask_rcnn = cv2.imread(os.path.join(self.mask_rcnn_dir, video, "%05d"%(frame_no) + ".png"), 0)
        stm = cv2.imread(os.path.join(self.stm_dir, video, "%05d"%(frame_no) + ".png"), 0)
        gt_img = cv2.imread(os.path.join(self.gt_dir, video, "%05d"%(frame_no) + ".png"), 0)
        curr_mask_rcnn = np.uint8(mask_rcnn == object_no)
        curr_stm = np.uint8(stm == object_no)
        mask_ind = np.where(curr_mask_rcnn)
        stm_ind = np.where(curr_stm)

        m_num = float(np.sum(np.logical_and(gt_img == object_no, curr_mask_rcnn == 1)))
        m_denom = float(np.sum(np.logical_or(gt_img == object_no, curr_mask_rcnn ==1)))
        m_iou = m_num/m_denom

        s_num = float(np.sum(np.logical_and(gt_img == object_no, curr_stm == 1)))
        s_denom = float(np.sum(np.logical_or(gt_img == object_no, curr_stm==1)))
        s_iou = s_num/s_denom

        mask_box = np.zeros(4, dtype=np.int32)
        mask_box[0] = int(np.min(mask_ind[0]))
        mask_box[1] = int(np.min(mask_ind[1]))
        mask_box[2] = int(np.max(mask_ind[0]))
        mask_box[3] = int(np.max(mask_ind[1]))

        stm_box = np.zeros(4, dtype=np.int32)
        stm_box[0] = int(np.min(stm_ind[0]))
        stm_box[1] = int(np.min(stm_ind[1]))
        stm_box[2] = int(np.max(stm_ind[0]))
        stm_box[3] = int(np.max(stm_ind[1]))

        final_box = mask_box

        if stm_box[0] < mask_box[0]:
            final_box[0] = stm_box[0]
        if stm_box[1] < mask_box[1]:
            final_box[1] = stm_box[1]
        if stm_box[2] > mask_box[2]:
            final_box[2] = stm_box[2]
        if stm_box[3] > mask_box[3]:
            final_box[3] = stm_box[3]


        mask_crop = cv2.resize(curr_mask_rcnn, (default_width, default_height), interpolation=cv2.INTER_NEAREST)
        stm_crop = cv2.resize(curr_stm, (default_width, default_height), interpolation=cv2.INTER_NEAREST)
        rgb_crop = cv2.resize(rgb_img, (default_width, default_height))

        mask_crop = np.reshape(mask_crop, (mask_crop.shape[0], mask_crop.shape[1], 1))
        stm_crop = np.reshape(stm_crop, (stm_crop.shape[0], stm_crop.shape[1], 1))
        mask_input = np.concatenate((rgb_crop, mask_crop), -1)
        stm_input = np.concatenate((rgb_crop, stm_crop), -1)

        input_dict = {}
        label_c = 1.
        label_f = 0.
        if random.random() >= 0.5:
            input_dict["input1"] = torch.from_numpy(np.transpose(mask_input, (2, 0, 1))).float()
            input_dict["input2"] = torch.from_numpy(np.transpose(stm_input, (2, 0, 1))).float()
            if s_iou < m_iou:
                input_dict["output1"] = label_c
                input_dict["output2"] = label_f
            else:
                input_dict["output1"] = label_f
                input_dict["output2"] = label_c               
        else:
            input_dict["input2"] = torch.from_numpy(np.transpose(mask_input, (2, 0, 1))).float()
            input_dict["input1"] = torch.from_numpy(np.transpose(stm_input, (2, 0, 1))).float()
            if s_iou < m_iou:
                input_dict["output1"] = label_f
                input_dict["output2"] = label_c
            else:
                input_dict["output1"] = label_c
                input_dict["output2"] = label_f

        return input_dict        

if __name__ == '__main__':
    pass