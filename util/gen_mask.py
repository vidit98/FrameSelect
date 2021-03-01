import argparse
import pathlib

import detectron2
import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import detection_utils
from PIL import Image

RCNN_THRESHOLD = 0.1
MAX_DETECT = 10


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description='Extract masks from dataset.')
    parser.add_argument('dataset_path', type=pathlib.Path)
    parser.add_argument('dest_path', type=pathlib.Path)
    parser.add_argument(
        '--model',
        default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    args = parser.parse_args()
    extract(args.dataset_path, args.dest_path, args.model)


def extract(dataset_path, dest_path, model):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = RCNN_THRESHOLD
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)
    input_frames = dataset_path.glob('*.jpg')
    if input_frames:
        dest_path.mkdir(parents=True, exist_ok=True)
    for input_frame in input_frames:
        im = detection_utils.read_image(str(input_frame))
        outputs = predictor(im)
        instances = outputs['instances']
        # get the top X predicted object masks
        masks = outputs['instances'].pred_masks[:MAX_DETECT]
        # re-sort biggest to largest to prevent small masks from being occluded
        # by large masks
        areas = np.asarray([m.sum().cpu() for m in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        out = np.zeros(im.shape[:2], dtype=np.uint8)
        for ind in range(len(masks)):
            m = masks[sorted_idxs[ind]]
            mask_ind = np.nonzero(np.array(m.cpu()))
            # 0 is the background value, so start numbering from 1
            out[mask_ind] = ind + 1
        out_im = Image.fromarray(out)
        out_im.save(str(dest_path / input_frame.name))


if __name__ == '__main__':
    main()
