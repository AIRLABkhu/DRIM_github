import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, image_dir, gt_dir, inter_dir, sensor_dir, mode, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.inter_dir = inter_dir
        self.sensor_dir = sensor_dir
        self.mode = mode

        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.gt_files = [f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))]
        self.inter_files = [f for f in os.listdir(inter_dir) if os.path.isfile(os.path.join(inter_dir, f))]
        self.sensor_files = [f for f in os.listdir(sensor_dir) if os.path.isfile(os.path.join(sensor_dir, f))]

        self.image_files.sort()
        self.gt_files.sort()
        self.inter_files.sort()
        self.sensor_files.sort()

        assert len(self.image_files) == len(self.gt_files) == len(self.inter_files) == len(self.sensor_files), "The number of images and gts should be the same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        inter_path = os.path.join(self.inter_dir, self.inter_files[idx])
        sensor_path = os.path.join(self.sensor_dir, self.sensor_files[idx])

        image = Image.open(img_path)
        gt = Image.open(gt_path)
        inter = Image.open(inter_path)
        sensor = Image.open(sensor_path)

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)
            inter = transforms.ToTensor()(inter)
            sensor = transforms.ToTensor()(sensor)

        mask = torch.zeros_like(inter).long()
        mask[inter == 0] = 1
        mask[sensor == 0] = 2

        return image, gt, mask.squeeze() #inter, sensor

class RandomHorizontalFlipWithMask:

    def __call__(self, image, gt, inter, sensor):
        return transforms.functional.hflip(image), transforms.functional.hflip(gt), transforms.functional.hflip(inter), transforms.functional.hflip(sensor)


class RandomVerticalFlipWithMask:

    def __call__(self, image, gt, inter, sensor):
        return transforms.functional.vflip(image), transforms.functional.vflip(gt), transforms.functional.vflip(inter), transforms.functional.vflip(sensor)


class RandomScalingWithMask:
    def __init__(self):

        self._scale = np.random.uniform(1.0, 1.5)
        self.scale = int(480 * self._scale)
        self.trans = transforms.Compose([transforms.Resize(self.scale),
                                    transforms.CenterCrop((480,640))])

    def __call__(self, image, gt, inter, sensor):

        image = self.trans(image)
        gt = self.trans(gt)
        inter = self.trans(inter)
        sensor = self.trans(sensor)

        return image, gt, inter, sensor


class RandomRotateWithMask:
    def __init__(self):
        self.degree = np.random.uniform(-5.0, 5.0)

    def __call__(self, image, gt, inter, sensor):
        image = transforms.functional.rotate(image, angle=self.degree)
        gt = transforms.functional.rotate(gt, angle=self.degree)
        inter = transforms.functional.rotate(inter, angle=self.degree, fill=255)
        sensor = transforms.functional.rotate(sensor, angle=self.degree, fill=0)
        return image, gt, inter, sensor


class CenterCrop:
    
    def __call__(self, image, gt, inter, sensor):    
        image_org = np.array(image)
        gt_org = np.array(gt)
        inter_org = np.array(inter)
        sensor_org = np.array(sensor)
        
        _scale = np.random.uniform(0.8, 1.0)
        scale = int(480 * _scale)
        scale2 = int(640 * _scale)

        t_dep = transforms.Compose([transforms.CenterCrop((scale,scale2))])

        image = t_dep(image)
        image = np.array(image)

        gt = t_dep(gt)
        gt = np.array(gt)

        inter = t_dep(inter)
        inter = np.array(inter)

        sensor = t_dep(sensor)
        sensor = np.array(sensor)

        # 센터 크롭 부분을 0으로 초기화된 텐서에 복사
        crop_h_start = (image_org.shape[0] - scale) // 2
        crop_w_start = (image_org.shape[1] - scale2) // 2

        image_org = np.zeros_like(image_org)
        image_org[crop_h_start:crop_h_start + scale, crop_w_start:crop_w_start + scale2] = image
        image = image_org

        gt_org = np.zeros_like(gt_org)
        gt_org[crop_h_start:crop_h_start + scale, crop_w_start:crop_w_start + scale2] = gt
        gt = gt_org

        inter_org = np.ones_like(inter_org)
        inter_org = inter_org * 255
        inter_org[crop_h_start:crop_h_start + scale, crop_w_start:crop_w_start + scale2] = inter
        inter = inter_org

        sensor_org = np.zeros_like(sensor_org)
        sensor_org[crop_h_start:crop_h_start + scale, crop_w_start:crop_w_start + scale2] = sensor
        sensor = sensor_org

        # NumPy 배열을 이미지로 변환
        image = Image.fromarray(image.astype(np.uint16))
        gt = Image.fromarray(gt.astype(np.uint16))
        inter = Image.fromarray(inter.astype(np.uint8))
        sensor = Image.fromarray(sensor.astype(np.uint8))
        return image, gt, inter, sensor


class RandomErase(object):
    def __init__(self, probability=1.0, scale=(0.001, 0.05), ratio=(0.3, 3.3)):
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img, gt, inter, sensor):
        return self._random_erase(img, gt, inter, sensor)

    def _random_erase(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)
        height, width = img_array.shape[:2]

        # Calculate area and aspect ratio
        area = width * height
        target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

        h = int(np.round(np.sqrt(target_area * aspect_ratio)))
        w = int(np.round(np.sqrt(target_area / aspect_ratio)))

        # Ensure that the width and height are within bounds
        if w <= width and h <= height:
            top = np.random.randint(0, height - h + 1)
            left = np.random.randint(0, width - w + 1)

            # Apply random erase
            img_array[top:top+h, left:left+w] = 0
            gt_array[top:top+h, left:left+w] = 0
            inter_array[top:top+h, left:left+w] = 255
            sensor_array[top:top+h, left:left+w] = 0

        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))
        return image, gt, inter, sensor

class RandomEraseEdges(object):
    def __init__(self, erase_ratio=0.5, patch_size=5):
        self.erase_ratio = erase_ratio
        self.patch_size = patch_size

    def __call__(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)

        gt_array_org = np.array(gt)
        gt_array_org = (((gt_array_org-gt_array_org.min()))/(gt_array_org.max() - gt_array_org.min())*255).astype(np.uint8)
        
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)
        height, width = img_array.shape[:2]

        # Step 1: Edge detection using Canny
        edges = cv2.Canny(gt_array_org, threshold1=50, threshold2=150)

        # Step 2: Find edge pixels
        edge_pixels = np.argwhere(edges > 0)

        num_pixels_to_erase = int(len(edge_pixels) * self.erase_ratio)
        if num_pixels_to_erase > 0:
            selected_pixels = edge_pixels[np.random.choice(len(edge_pixels), num_pixels_to_erase, replace=False)]

            # Step 4: Erase the selected edge pixels with patches
            for (y, x) in selected_pixels:
                top_left_y = max(0, y - self.patch_size // 2)
                top_left_x = max(0, x - self.patch_size // 2)
                bottom_right_y = min(height, y + self.patch_size // 2 + 1)
                bottom_right_x = min(width, x + self.patch_size // 2 + 1)
                
                # Apply the patch erase
                img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
                gt_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
                inter_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
                sensor_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
        
        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))

        return image, gt, inter, sensor

class RandomEraseFromEdge(object):
    def __init__(self, scale=(0.001, 0.05), ratio=(0.3, 3.3)):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)

        gt_array_org = np.array(gt)
        gt_array_org = (((gt_array_org-gt_array_org.min()))/(gt_array_org.max() - gt_array_org.min())*255).astype(np.uint8)
        
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)
        height, width = img_array.shape[:2]
        
        # Step 1: Edge detection using Canny
        edges = cv2.Canny(gt_array_org, threshold1=50, threshold2=150)

        # Step 2: Find edge pixels
        edge_pixels = np.argwhere(edges > 0)
        
        # Step 3: Randomly select one edge pixel
        if len(edge_pixels) > 0:
            y, x = edge_pixels[random.randint(0, len(edge_pixels) - 1)]

            # Step 4: Calculate random patch size
            area = width * height
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

            h = int(np.round(np.sqrt(target_area * aspect_ratio)))
            w = int(np.round(np.sqrt(target_area / aspect_ratio)))

            # Ensure the patch size fits within the image boundaries
            top_left_y = max(0, y - h // 2)
            top_left_x = max(0, x - w // 2)
            bottom_right_y = min(height, y + h // 2 + 1)
            bottom_right_x = min(width, x + w // 2 + 1)

            # Step 5: Apply random erase
            img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
            gt_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
            inter_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
            sensor_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
        
        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))

        return image, gt, inter, sensor
    
class DiagonalRegionErase(object):
    def __init__(self, image_size=(480, 640)):
        self.image_size = image_size
        self.height, self.width = image_size

    def __call__(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)

        # Step 1: Select one random pixel from each corner
        top_pixel = (0, random.randint(0, self.height - 1))           # Top edge
        bottom_pixel = (self.width - 1, random.randint(0, self.height - 1))  # Bottom edge
        left_pixel = (random.randint(0, self.width - 1), 0)          # Left edge
        right_pixel = (random.randint(0, self.width - 1), self.height - 1)   # Right edge

        # Step 2: Create a mask that will keep only the region inside the diagonals
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)

        # Draw the polygon formed by the intersection of the edges
        draw.polygon([top_pixel, right_pixel, bottom_pixel, left_pixel], fill=255)
        
        # Convert the mask to a numpy array
        mask_array = np.array(mask)

        # Step 3: Apply the mask to the image
        img_array[mask_array == 0] = 0  # Set pixels outside the polygon region to 0
        gt_array[mask_array == 0] = 0
        inter_array[mask_array == 0] = 255
        sensor_array[mask_array == 0] = 0

        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))

        return image, gt, inter, sensor


class DepthDataset_aug(Dataset):
    def __init__(self, image_dir, gt_dir, inter_dir, sensor_dir, mode, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.inter_dir = inter_dir
        self.sensor_dir = sensor_dir
        self.mode = mode

        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.gt_files = [f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))]
        self.inter_files = [f for f in os.listdir(inter_dir) if os.path.isfile(os.path.join(inter_dir, f))]
        self.sensor_files = [f for f in os.listdir(sensor_dir) if os.path.isfile(os.path.join(sensor_dir, f))]

        self.image_files.sort()
        self.gt_files.sort()
        self.inter_files.sort()
        self.sensor_files.sort()

        assert len(self.image_files) == len(self.gt_files) == len(self.inter_files) == len(self.sensor_files), "The number of images and gts should be the same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        inter_path = os.path.join(self.inter_dir, self.inter_files[idx])
        sensor_path = os.path.join(self.sensor_dir, self.sensor_files[idx])

        image = Image.open(img_path)
        gt = Image.open(gt_path)
        inter = Image.open(inter_path)
        sensor = Image.open(sensor_path)

        if random.random() < 0.5:
            
            random_num = random.random()

            if random_num < 0.1111:
                image, gt, inter, sensor = RandomHorizontalFlipWithMask()(image, gt, inter, sensor)
            elif 0.1111 <= random_num and random_num < 0.2222:
                image, gt, inter, sensor = RandomVerticalFlipWithMask()(image, gt, inter, sensor)
            elif 0.2222 <= random_num and random_num < 0.3333:
                image, gt, inter, sensor = RandomRotateWithMask()(image, gt, inter, sensor)
            elif 0.3333 <= random_num and random_num < 0.4444:
                image, gt, inter, sensor = CenterCrop()(image, gt, inter, sensor)
            elif 0.4444 <= random_num and random_num < 0.5555:
                image, gt, inter, sensor = DiagonalRegionErase()(image, gt, inter, sensor)
            elif 0.5555 <= random_num and random_num < 0.6666:
                image, gt, inter, sensor = RandomErase()(image, gt, inter, sensor)
            elif 0.6666 <= random_num and random_num < 0.7777:
                image, gt, inter, sensor = RandomEraseFromEdge()(image, gt, inter, sensor)
            elif 0.7777 <= random_num and random_num < 0.8888:
                image, gt, inter, sensor = RandomEraseEdges()(image, gt, inter, sensor)
    
    
        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)
            inter = transforms.ToTensor()(inter)
            sensor = transforms.ToTensor()(sensor)

        mask = torch.zeros_like(inter).long()
        mask[inter == 0] = 1
        mask[sensor == 0] = 2

        return image, gt, mask.squeeze() #inter, sensor
