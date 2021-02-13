import numpy as np
import argparse
import os
import itertools
import cv2
from scipy.optimize import linear_sum_assignment
import glob
from sn_utils import process_single_image

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

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-m1", type=str, help="path to criteria 1 masks folder", required=True)
    parser.add_argument("-m2", type=str, help="path to criteria 2 masks folder", required=True)
    parser.add_argument("-r", type=str, help="path to rgb data folder",required=True)
    parser.add_argument("-f", type=str, help="set file containing all videos name",required=True)
    parser.add_argument("-s", type=str, help="output directory",default='./output')
    return parser.parse_args()

args = get_arguments()

sets_file = args.f
vids = [x.strip() for x in open(sets_file).readlines()]

mask_image1_dir = args.m1
mask_image2_dir = args.m2
rgb_img_dir = args.r
save_dir = args.s

for vid in vids:
  print(vid)

  paths = sorted(glob.glob(os.path.join(mask_image1_dir, vid, "*.png")))
  i = 0
  for path in paths:
    basename = os.path.basename(path)

    #criteria 1 mask
    img1 = cv2.imread(path, 0)

    #criteria 2 mask
    img2 = cv2.imread(os.path.join(mask_image2_dir, vid, basename), 0)

    #rgb image
    rgb_img = cv2.imread(os.path.join(rgb_img_dir, vid, basename[:-3]+"jpg"), 1)

    save_img = np.zeros_like(img1)

    #finding association b/w both the masks object
    assoc_r, assoc_c = calc_intersect(img1, img2)
    

    unique_m = np.unique(img1[img1!=0])
    unique_l = np.unique(img2[img2!=0])


    if i == 0:
      no_of_obj = np.max(unique_m)
      print("number of objects are " + str(no_of_obj))

    for m in unique_m:

      #find out the better mask for overlapping object
      if m in assoc_r:
        ind1 = assoc_r.index(m)
        col2 = assoc_c[ind1]
        prob = process_single_image(rgb_img, np.uint8(img1==m), np.uint8(img2==col2))

        if prob[0] > prob[1]:
          save_img[img2==col2] = m
        else:
          save_img[img1==m] = m
      else:   #add missing objects directly
        save_img[img1==m] = m


    #add missing objects directly
    for l in unique_l:
       if l not in assoc_c:
         if l <= no_of_obj:
          save_img[img2==l] = l

    if not os.path.exists(os.path.join(save_dir, vid)):
      os.makedirs(os.path.join(save_dir, vid))

    cv2.imwrite(os.path.join(save_dir, vid, basename), save_img)
    i = i + 1