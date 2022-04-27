# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def test_time(model):
    all_time_pred = []
    all_time_nms = []
    for i in tqdm.tqdm(range(100)):
        image = torch.randint(0,255,(8,3, 640, 640), dtype=torch.float32)
        image = image/255

        t = time.time()

        pred = model(image)[0]

        t1 = time.time()-t

        t = time.time()
        pred = non_max_suppression_face(pred, 0.3, 0.5)
        t2 = time.time()-t
        all_time_pred.append(t1)
        all_time_nms.append(t2)
    print(pred)
    print("Average time for prediction: ", sum(all_time_pred)/len(all_time_pred))
    print("Average time for NMS: ", sum(all_time_nms)/len(all_time_nms))
    print("Total time: ", (sum(all_time_pred)+sum(all_time_nms))/len(all_time_pred))


import glob

all_model_path = glob.glob('/Users/miles/Downloads/yolo_face_detection/*.pt')
for model_path in all_model_path:
    print(model_path)
    model = load_model(model_path, select_device())
    test_time(model)