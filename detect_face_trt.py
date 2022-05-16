# -*- coding: UTF-8 -*-
import argparse
import atexit
import copy
import os
import sys
import time
from pathlib import Path

import cv2
import imgaug.augmenters as iaa
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (apply_classifier, check_img_size, increment_path,
                           non_max_suppression_face, scale_coords, set_logging,
                           strip_optimizer, xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, select_device, time_synchronized

root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径
from detect_face import scale_coords_landmarks, show_results
from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression_face,
                           scale_coords, xyxy2xywh)

cur_path=os.path.abspath(os.path.dirname(__file__))

def img_vis(img,orgimg,pred,vis_thres = 0.6):
    '''
    预测可视化
    vis_thres: 可视化阈值
    '''
    import ipdb; ipdb.set_trace(context=10)

    no_vis_nums=0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                
                
                if det[j, 4].cpu().numpy() < vis_thres:
                    no_vis_nums+=1
                    continue

                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

    cv2.imwrite(cur_path+'/result.jpg', orgimg)
    print('result save in '+cur_path+'/result.jpg')


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)

    
class TRTModel(object):
    def __init__(self, engine_path, input_names=None, output_names=None, final_shapes=None):
        # load engine
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names
            
        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names
            
        self.final_shapes = final_shapes
        import ipdb; ipdb.set_trace(context=10)
        # destroy at exit
        atexit.register(self.destroy)
    
    def _input_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
    
    def _output_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
    
    def _trt_input_names(self):
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]
    
    def _trt_output_names(self):
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]
    
    def create_output_buffers(self, batch_size):
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size, ) + self.final_shapes[i]
            else:
                shape = (batch_size, ) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
        return outputs
    
    def execute(self, *inputs):
        batch_size = inputs[0].shape[0]
        
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        # map input bindings
        inputs_torch = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            
            # convert to appropriate format
            if not isinstance(inputs[i], torch.Tensor):
                inputs_torch[i] = torch.from_numpy(inputs[i])
            else:
                inputs_torch[i] = inputs[i]
            inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)))
            inputs_torch[i] = inputs_torch[i].type(torch_dtype_from_trt(self.engine.get_binding_dtype(idx)))
            
            bindings[idx] = int(inputs_torch[i].data_ptr())
        
        output_buffers = self.create_output_buffers(batch_size)
        
        # map output bindings
        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            bindings[idx] = int(output_buffers[i].data_ptr())
        
        self.context.execute(batch_size, bindings)

        # outputs = [buffer.cpu().numpy() for buffer in output_buffers]
        outputs = output_buffers

        return outputs
    
    def __call__(self, *inputs):
        return self.execute(*inputs)

    def destroy(self):
        del self.runtime
        del self.logger
        del self.engine
        del self.context

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def preprocess_batch(images,
                     target_size: int = -1, device = 'cpu'):
        # Resize on the longer side of the image to the target size (keep aspect ratio)
        batch_size = len(images)
        
        resized_images= []
        all_height = []
        all_width = []
        all_scale = []
        all_index = []

        image_size_dict = {}
        for idx, image in enumerate(images):
            height, width, _ = image.shape
            _size = f"{height}x{width}"
            if _size not in image_size_dict:
                image_size_dict[_size] = [idx]
            else:
                image_size_dict[_size].append(idx)
        
        # Dynamic batching
        for key in image_size_dict.keys():
            # image_height, image_width = list(map(int, key.split('x')))
            image_height, image_width = int(key.split('x')[0]), int(key.split('x')[1])
            scale: float = 1.
            batch = torch.stack([images[idx] for idx in image_size_dict[key]]).to(device).permute(0, 3, 1, 2).float()
            if target_size != -1:
                if max(image_height, image_width) > target_size:
                    if image_height >= image_width:
                        scale = target_size / image_height
                        new_height = target_size
                        new_width = int(image_width * scale)
                    else:
                        scale = target_size / image_width
                        new_width = target_size
                        new_height = int(image_height * scale)

                    resized_batch = F.interpolate(batch, size=(
                        new_height, new_width), mode="bicubic", align_corners=False)
                else:
                    new_height = image_height
                    new_width = image_width
                    resized_batch = batch
            else:
                new_height = image_height
                new_width = image_width
                resized_batch = batch

            resized_batch = resized_batch/255.0

            resized_images.extend(list(resized_batch))
            all_index.extend(image_size_dict[key])
            all_width.append(new_width)
            all_height.append(new_height)
            all_scale.extend([scale] * len(image_size_dict[key]))

        # zip(all_index, resized_images)
        temp = []
        for idx in range(len(all_index)):
            temp.append((all_index[idx], resized_images[idx]))
 
        # zip(all_index, all_scale)
        temp2 = []
        for idx in range(len(all_index)):
            temp2.append((all_index[idx], all_scale[idx]))

        resized_images = [x for _, x in sorted(temp)]
        all_scale = [x for _, x in sorted(temp2)]

        # Zero padding sequential
        # max_width = max(all_width)
        # max_height = max(all_height)
        batched_tensor = torch.zeros((batch_size, 3, target_size, target_size), device=device).float()
        for index in range(batch_size):
            resized_image = resized_images[index]
            image_size = resized_image.size()
            image_height, image_width = int(image_size[1]), int(image_size[2])
            height_offset = (target_size - image_height) // 2
            width_offset = (target_size - image_width) // 2
            if height_offset > width_offset:
                batched_tensor[index, :, height_offset:height_offset + image_height, :] = resized_image
            else:
                batched_tensor[index, :, :, width_offset:width_offset + image_width] = resized_image
            # batiaached_tensor[index, :, :image_height,
            #                :image_width] = resized_image
        return batched_tensor, all_scale

def detect_batch(model,
                list_image, device, target_size=640, name=""):
    img_size = target_size
    conf_thres = 0.6
    iou_thres = 0.4
    original_list = list_image.copy()
    output_list = []
    for image in list_image:
        output_list.append(torch.tensor(image))
    
    st = time.time()
    batched_tensor, all_scale = preprocess_batch(output_list, img_size, device)
    # batched_tensor = img_process(list_image[0])
    print(f"Preprocess time: {(time.time()-st)*1000:.2f}ms")

    st = time.time()
    pred0 = model.execute(batched_tensor)[0]
    print(f"Inference time: {(time.time()-st)*1000:.2f}ms")
    pred = pred0[0]

    st = time.time()
    import ipdb; ipdb.set_trace(context=10)
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    print(f"Postprocess time: {(time.time()-st)*1000:.2f}ms")

    img_vis(batched_tensor, original_list[0], [item.cpu() for item in pred])
    # output_list = []
    # try:
    #     for index, image in enumerate(list_image):  # detections per image
    #         det = pred[index]
    #         bbox = (det[:,:4]/all_scale[index]).round()
    #         conf = det[:,4]
    #         landmark = (det[:,5:15]/all_scale[index]).round()

    #         for j in range(det.size()[0]):
    #             x1,y1, x2,y2 = list(map(int,bbox[j,:].tolist()))
    #             cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
    #             landmarks = (landmark[j].view(1, 10)).view(-1).tolist()
    #             for k in range(0,10,2):
    #                 cv2.circle(image, (int(landmarks[k]),int(landmarks[k+1])), 1, (0,0,255), 2)
    #             # class_num = det[j, 15].cpu().numpy()
    #         cv2.imwrite(f'./debug/{name}_{index}.jpg', image)
    #         output_list.append(image)
    # except:
    #     import traceback
    #     traceback.print_exc()

    return output_list

seq = iaa.Sequential([
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
], random_order=True) # apply augmenters in random order


if __name__=="__main__":
    device = torch.device('cuda')
    model = TRTModel('yolov5n-0.5.trt')
    # model = TRTModel('yolov5n-0.5_b4_genos.trt')
    # model = TRTModel('yolov5n-0.5-trtexec.trt')

    path = ["/home/yolov5-face/data/images/test.jpg",
            "data/images/zidane.jpg"]
    batch_size = 4
    for bs in range(1,1000):
        bs = 4
        print(f"#"*10 + f" batch size {bs} " + f"#"*10)
        images = [cv2.imread(np.random.choice(path)) for _ in range(bs)]
        list_image = seq.augment(images=images)
        try:
            output_list = detect_batch(model,list_image, device, target_size=640, name=bs)
            torch.cuda.empty_cache()
        except:
            print(f"Error with batch size {bs}")

    # detect_batch(model,[cv2.imread(path[0])], device, target_size=640, name="TEST")
