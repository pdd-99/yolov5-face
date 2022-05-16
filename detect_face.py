# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import torch
import copy
import torch.nn.functional as F

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import imgaug.augmenters as iaa

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
            batched_tensor[index, :, :image_height,
                           :image_width] = resized_image
        return batched_tensor, all_scale

def detect_one(model, orgimg, device, target_size=480):
    # Load model
    img_size = target_size
    conf_thres = 0.3
    iou_thres = 0.5

    # orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'fuck you'
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t1 = time_synchronized()
    t0 = time.time()
    pred = model(img)[0]
    print(f"Single inference time: {(time.time()-t0)*1000:.2f}ms")
    t0= time.time()
    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    print(f"Single postprocessing time: {(time.time()-t0)*1000:.2f}ms")

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

    return orgimg

def detect_batch(model,
                list_image, device, target_size=640):
    img_size = target_size
    conf_thres = 0.3
    iou_thres = 0.5

    output_list = []
    for image in list_image:
        output_list.append(torch.tensor(image))
    
    st = time.time()
    batched_tensor, all_scale = preprocess_batch(output_list, img_size, device)
    print(f"Preprocess time: {(time.time()-st)*1000:.2f}ms")

    st = time.time()
    pred = model(batched_tensor)[0]
    print(f"Inference time: {(time.time()-st)*1000:.2f}ms")

    st = time.time()
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    print(f"Postprocess time: {(time.time()-st)*1000:.2f}ms")
    
    output_list = []
    for i, image in enumerate(list_image):  # detections per image
        det = pred[i]
        bbox = (det[:,:4]/all_scale[i]).round()
        conf = det[:,4]
        landmark = (det[:,5:15]/all_scale[i]).round()

        for j in range(det.size()[0]):
            x1,y1, x2,y2 = list(map(int,bbox[j,:].tolist()))
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            landmarks = (landmark[j].view(1, 10)).view(-1).tolist()
            for k in range(0,10,2):
                cv2.circle(image, (int(landmarks[k]),int(landmarks[k+1])), 1, (0,0,255), 2)
            # class_num = det[j, 15].cpu().numpy()
        cv2.imwrite(f'./debug/{i}.jpg', image)
        output_list.append(image)

    return output_list

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
], random_order=True) # apply augmenters in random order

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5n-0.5.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    device = torch.device('cuda')
    model = load_model(opt.weights, device)

    path = "/home/yolov5-face/data/images/test.jpg"
    # path = "data/images/zidane.jpg"
    image = cv2.imread(path)
    # for _ in range(10):
    list_image = seq(images=[image]*4)
    list_image = [image]
    output_list = detect_batch(model,list_image, device, target_size=640)
    # for i, image in enumerate(output_list):
    #     cv2.imwrite(f'./debug/{i}.jpg', image)
    print(f"#"*15)