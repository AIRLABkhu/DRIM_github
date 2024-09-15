import os
import numpy as np
from stream_metrics import StreamSegMetrics
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image



from model import *
from dataset import *
from utils import *

import warnings
warnings.filterwarnings('ignore')


def evaluate(model, dataloader, device, batch_size, metrics, exp=None, save_img=False):
    metrics.reset()
    model.eval()
    test_output_dir = 'eval_output'
    os.makedirs(os.path.join(exp, test_output_dir), exist_ok=True)

    all_rmse = []
    all_psnr = []
    
    with torch.no_grad():
        for i, (images, gts, masks) in enumerate(tqdm.tqdm(dataloader)):
            images, gts, masks = images.to(device), gts.to(device), masks.to(device)
            outputs, mask_pred = model(images)
            outputs = torch.clamp(outputs, min=-1., max=1.)

            probs = torch.softmax(mask_pred, dim=1)
            mask = torch.argmax(probs, dim=1).unsqueeze(1)
      
            metrics.update(masks.cpu().numpy(), mask_pred.detach().max(dim=1)[1].cpu().numpy())

            inter_mask = torch.zeros_like(mask).float()
            inter_mask[mask==1] = 1
            sensor_mask = torch.zeros_like(mask).float()
            sensor_mask[mask==2] = 1
            
            images = denormalize(images)
            outputs = denormalize(outputs)
            gts = denormalize(gts)
            
            batch_rmse = rmse(outputs, gts).cpu().numpy()
            batch_psnr = psnr(outputs, gts).cpu().numpy()

            all_rmse.extend(batch_rmse)
            all_psnr.extend(batch_psnr)
            
            if save_img:
                gt_inter_mask = torch.zeros_like(masks).float()
                gt_inter_mask[masks==1] = 1
                gt_inter_mask = gt_inter_mask.unsqueeze(1)
                gt_sensor_mask = torch.zeros_like(masks).float()
                gt_sensor_mask[masks==2] = 1
                gt_sensor_mask = gt_sensor_mask.unsqueeze(1)

                for j in range(outputs.size(0)):
                    save_depth(outputs[j].detach().cpu(), os.path.join(os.path.join(exp, test_output_dir), f'output_{i*batch_size+j:05d}.png'))
                    combined_outputs = torch.cat([minmax(images[j].detach().cpu()), minmax(outputs[j].detach().cpu()), minmax(gts[j].detach().cpu())], dim=2)
                    combined_masks = torch.cat([inter_mask[j].detach().cpu(), sensor_mask[j].detach().cpu(), gt_inter_mask[j].detach().cpu(), gt_sensor_mask[j].detach().cpu()], dim=2)

                    save_image(combined_outputs, os.path.join(os.path.join(exp, test_output_dir), f'combined_outputs_{i*batch_size+j:05d}.png'), normalize=False)
                    save_image(combined_masks, os.path.join(os.path.join(exp, test_output_dir), f'combined_masks_{i*batch_size+j:05d}.png'))
                    
    avg_rmse = np.mean(all_rmse)
    avg_psnr = np.mean(all_psnr)
    score = metrics.get_results()

    return avg_rmse, avg_psnr, score

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiments Setting
    parser.add_argument('--exp', type=str, default='psnr', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Model Setting
    parser.add_argument('--model_name', type=str, default='unet', choices=['unet', 'DeepLabV3Plus'], help='model name')
    parser.add_argument('--encoder_name', type=str, default='resnet34', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='encoder name')
    # Datasets
    parser.add_argument('--image_dir', type=str, default='/DRIM/raw_dataset/trainsets/depth')
    parser.add_argument('--gt_dir', type=str, default='/DRIM/raw_dataset/trainsets/gt')
    parser.add_argument('--inter_mask_dir', type=str, default='/DRIM/raw_dataset/trainsets/inter_artifact_mask')
    parser.add_argument('--sensor_mask_dir', type=str, default='/DRIM/raw_dataset/trainsets/sensor_artifact_mask')
    # Training Setting
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--interval', type=int, default=100, help='image save interval')
    
    args = parser.parse_args()

    exp = os.path.join('/data/yoon2926/repos/DC/experiments', args.exp)
    print(exp)
    print(f"lr: {args.lr}")
    os.makedirs(exp, exist_ok=True)
    
    with open(os.path.join(exp, "args.txt"), "w") as f:
        for arg in vars(args).items():
            f.write(f"{arg}\n")
            
    seed_torch(seed=args.seed)
    device = torch.device("cuda")
    
    img_dir = args.image_dir
    gt_dir = args.gt_dir
    inter_dir = args.inter_mask_dir       
    sensor_dir = args.sensor_mask_dir      

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 65535.),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    test_dataset = DepthDataset(img_dir+'/test', gt_dir+'/test', inter_dir+'/test', sensor_dir+'/test', mode = 'test', transform=transform)
    
    print(len(test_dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DRIM().to(device)
    
    print("DRIM : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.load_state_dict(torch.load(os.path.join(exp, 'checkpoints', f'best.pth')))
    
    metrics = StreamSegMetrics(3)
    test_rmse, test_psnr, test_score = evaluate(model, test_dataloader, device, args.batch_size, metrics, save_img=True, exp=exp)
    
    print(f'Test RMSE: {test_rmse:.6f}')
    print(f'Test PSNR: {test_psnr:.2f}')
    print()
    print(metrics.to_str(test_score))
