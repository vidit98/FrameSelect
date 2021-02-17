# Unsupervised Video Object Segmentation using Online Mask Selection and Space-time Memory Networks


README.md will be updated soon. 

## Introduction
[**WACV**](https://openaccess.thecvf.com/content/WACV2021/papers/Garg_Mask_Selection_and_Propagation_for_Unsupervised_Video_Object_Segmentation_WACV_2021_paper.pdf) [**CVPRW**](https://davischallenge.org/challenge2020/papers/DAVIS-Unsupervised-Challenge-1st-Team.pdf)

![](final.gif)

## Prerequisites

## Inferencing
To run the code you will be needing masks from Mask R-CNN and DAVIS dataset. The pre-trained model of STM can be downloaded from [here](https://github.com/seoungwugoh/STM). The output masks of Mask R-CNN should be numbered sequentially starting from 0 representing background. Place the masks in `path_to_data_dir/Annotations/480p` and the DAVIS frames in `path_to_data_dir/JPEGImages/480p`. There are 3 parts of the method `criterion 1` `criteria 2` `stage 3`. For running `criterion 1` `criteria 2`  using `run.sh` file as all videos might not fit in the memory at once.



