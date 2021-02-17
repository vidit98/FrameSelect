# Unsupervised Video Object Segmentation using Online Mask Selection and Space-time Memory Networks


README.md will be updated soon. 

## Introduction
[[**WACV**](https://openaccess.thecvf.com/content/WACV2021/papers/Garg_Mask_Selection_and_Propagation_for_Unsupervised_Video_Object_Segmentation_WACV_2021_paper.pdf)] [[**CVPRW**](https://davischallenge.org/challenge2020/papers/DAVIS-Unsupervised-Challenge-1st-Team.pdf)]

![](final.gif)

## Prerequisites

## Inferencing
To run the code you will be needing masks from Mask R-CNN and DAVIS dataset. The pre-trained model of STM can be downloaded from [here](https://github.com/seoungwugoh/STM) and selector_net from [here](https://github.com/seoungwugoh/STM). The output masks of Mask R-CNN should be numbered sequentially starting from 0 representing background. 

1. Place the masks in `path_to_data_dir/Annotations/480p` and the DAVIS frames in `path_to_data_dir/JPEGImages/480p`. 
2. Place the downloaded weights of STM and selecto_net in `checkpoints` folder. There are 3 parts of the method criterion 1, criteria 2, stage 3. 
3. Evaluate using criterion 1 by executing `run.sh`(change the datapath).
4. Change the python script name in `run.sh` to `eval_DAVIS_crit2.py` to evaluate using criterion 2. 
5. Finally, run `eval_stage_3.py` by giving relavent paths of masks generated using criterion 1 and criterion 2. 
### Note
Selector net has been trained using Mask R-CNN outputs and the outputs of Mask R-CNN can highly vary depending to various implementations and even different hyperparameter in the same implementation. Hence, to get the true results of the method it is desirable to train the selector_net using the object detection and segmentation network that you are using. Training selector_net should not take more than 1hr. Training details are given below.

To run the code on other datasets, change the structure of data same as DAVIS dataset then the same scripts can be used.

## Training


