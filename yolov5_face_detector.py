import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from yolov5_trt import TRTModel
from yolov5_utils import (img_vis, map_prediction_result_to_original,
                          non_max_suppression_face)


class Yolov5FaceDetector:
    def __init__(self, model_path: str="yolov5n-0.5.trt"):
        self.model_path = model_path
        self.model = TRTModel(model_path)

    # @torch.jit.script_method
    def preprocess_batch(self, 
                        images, 
                        target_size: int=640,
                        device: str='cuda:0',
                        max_batch_size: int=None):
        # Resize on the longer side of the image to the target size (keep aspect ratio)
        if max_batch_size is None:
            max_batch_size = len(images)
        
        resized_images= []
        all_height = []
        all_width = []
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

        # zip(all_index, resized_images)
        temp = []
        for idx in range(len(all_index)):
            temp.append((all_index[idx], resized_images[idx]))
 
        resized_images = [x for _, x in sorted(temp)]

        # Zero padding sequential
        # max_width = max(all_width)
        # max_height = max(all_height)
        batched_tensor = torch.zeros((max_batch_size, 3, target_size, target_size), device=device).float()
        for index in range(len(resized_images)):
            resized_image = resized_images[index]
            image_size = resized_image.size()
            image_height, image_width = int(image_size[1]), int(image_size[2])
            height_offset = (target_size - image_height) // 2
            width_offset = (target_size - image_width) // 2
            if height_offset > width_offset:
                batched_tensor[index, :, height_offset:height_offset + image_height, :] = resized_image
            else:
                batched_tensor[index, :, :, width_offset:width_offset + image_width] = resized_image
        return batched_tensor

    def reformat_postprocessed_batch(self, postprocessed_batch):
        # Remove the last value of each prediction.
        # Prediction shape is reformatted from [.,16] to [.,15]
        return [item[:,:-1] for item in postprocessed_batch]

    # @torch.jit.script_method
    def postprocess_batch(self,
                        pred: torch.Tensor,
                        conf_thres: float=0.6,
                        iou_thres: float=0.4,
                        preprocessed_batch: torch.Tensor=None,
                        original_images: List[torch.Tensor]=None
                        ) -> List[torch.Tensor]:
        postprocessed_batch = non_max_suppression_face(pred, conf_thres, iou_thres)
        reformatted_postprocessed_batch = self.reformat_postprocessed_batch(postprocessed_batch)
        return map_prediction_result_to_original(
            preprocessed_batch, original_images, reformatted_postprocessed_batch
        )

    def forward_batch(self,
                      images: List[torch.Tensor],
                      target_size: int=640,
                      conf_thres: float=0.6,
                      iou_thres: float=0.4,
                      max_batch_size: int=4,
                      device: str='cuda:0'):
        if len(images) > max_batch_size:
            raise Exception(f"Batch size {len(images)} is too large. Max batch size is {max_batch_size}")
        # NOTE: Target size for max length side
        # NOTE: max_batch_size is the batch size of converted tensorrt model. If we smh manage to 
        # find a way to support dynamic batch size, we can remove this limitation.
        # NOTE: Always batching with max_batch_size so tenssort model won't bug out.
        # Preprocessing
        # st = time.time()
        preprocessed_batch = self.preprocess_batch(images, target_size, device, max_batch_size)
        # torch.cuda.synchronize()
        # print("Preprocessing tooks:", (time.time()-st)*1000, "ms")

        # Forward pass
        # st = time.time()
        pred = self.model.execute(preprocessed_batch)
        # torch.cuda.synchronize()
        # print("Forward tooks:", (time.time()-st)*1000, "ms")

        # Postprocessing
        # st = time.time()
        pred = pred[0][0]
        if max_batch_size is not None:
            pred = pred[:len(images)]
            preprocessed_batch = preprocessed_batch[:len(images)]
        output = self.postprocess_batch(pred, conf_thres, iou_thres, preprocessed_batch, images)
        # torch.cuda.synchronize()
        # print("Postprocess tooks:", (time.time()-st)*1000, "ms")
        return output

import imgaug.augmenters as iaa

# Augment sequence to prevent model from doing some caching magic.
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
    max_batch_size = 4
    actual_batch_size = 2
    import cv2
    detector = Yolov5FaceDetector(model_path="yolov5n-0.5.trt")

    for _ in range(1):
        img = cv2.imread("data/images/test.jpg")
        list_img = seq.augment(images=[img]*actual_batch_size)
        original_list = list_img.copy()
        output = detector.forward_batch([torch.tensor(img) for img in  list_img], max_batch_size=max_batch_size)
        print("#"*30)

    for idx in range(len(output)):
        viz = img_vis(original_list[idx], output[idx])
        cv2.imwrite(f"yolov5_output_{idx}.jpg", viz)
