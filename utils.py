import random
import numpy as np
import cv2

import torch
import torch.nn.functional as F


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2, dim=(1, 2, 3)))

def psnr(predictions, targets, max_pixel_value=1.0):
    mse = torch.mean((predictions - targets) ** 2, dim=(1, 2, 3))
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value

def denormalize(image):
    return image * 0.5 + 0.5

def ssim(x, y, C1=0.01**2, C2=0.03**2):
    """Compute the SSIM between two images."""
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map) / 2, 0, 1)

def ssim_loss(x, y):
    """SSIM loss."""
    return ssim(x, y).mean()
    
def gradient_x(img):
    """Compute gradient along x-axis."""
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx

def gradient_y(img):
    """Compute gradient along y-axis."""
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy

def depth_smoothness_loss(pred, image):
    """Compute the smoothness loss for the predicted depth map."""
    pred_dx = gradient_x(pred)
    pred_dy = gradient_y(pred)

    image_dx = gradient_x(image)
    image_dy = gradient_y(image)

    weights_x = torch.exp(-torch.mean(torch.abs(pred_dx), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(pred_dy), 1, keepdim=True))

    smoothness_x = pred_dx * weights_x
    smoothness_y = pred_dy * weights_y

    return (smoothness_x.abs().mean() + smoothness_y.abs().mean())

def minmax(image):
    return (image - image.min()) / (image.max() - image.min())
    

def save_depth(depth, path):
    depth = depth * 65535
        
    np_image = depth.squeeze().numpy().astype(np.uint16)
    cv2.imwrite(path, np_image)
    
    return
