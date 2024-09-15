import os
import numpy as np
import tqdm
import time
import argparse
from stream_metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models
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
    test_output_dir = 'test_output'
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
    parser.add_argument('--exp', type=str, default='exp', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Datasets
    parser.add_argument('--image_dir', type=str, default='DRIM/raw_dataset/trainsets/depth')
    parser.add_argument('--gt_dir', type=str, default='DRIM/raw_dataset/trainsets/gt')
    parser.add_argument('--inter_mask_dir', type=str, default='DRIM/raw_dataset/trainsets/inter_artifact_mask')
    parser.add_argument('--sensor_mask_dir', type=str, default='DRIM/raw_dataset/trainsets/inter_artifact_mask')
    # Training Setting
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--interval', type=int, default=100, help='image save interval')
    
    args = parser.parse_args()

    exp = args.exp
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
    
    train_dataset = DepthDataset_aug(img_dir+'/train', gt_dir+'/train', inter_dir+'/train', sensor_dir + '/train', mode = 'train', transform=transform)
    val_dataset = DepthDataset(img_dir+'/valid', gt_dir+'/valid', inter_dir+'/valid', sensor_dir+'/valid', mode = 'valid', transform=transform)
    test_dataset = DepthDataset(img_dir+'/test', gt_dir+'/test', inter_dir+'/test', sensor_dir+'/test', mode = 'test', transform=transform)
    
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(train_dataset[0][0].shape)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    resnet_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101
    }

    model = DRIM().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    output_dir = 'output_images'
    os.makedirs(os.path.join(exp, output_dir), exist_ok=True)
    os.makedirs(os.path.join(exp, 'checkpoints'), exist_ok=True)
    
    start = time.time()
    # Training loop
    best_psnr = 0.0
    best_rmse = float('inf')
    best_epoch = 0
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0

        with tqdm.tqdm(train_dataloader) as pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            for i, (images, gts, masks) in enumerate(pbar):
                images, gts, masks = images.to(device), gts.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs, mask_pred = model(images)
                
                probs = torch.softmax(mask_pred, dim=1)
                mask = torch.argmax(probs, dim=1).unsqueeze(1)
          
                inter_mask = torch.zeros_like(mask).float()
                inter_mask[mask==1] = 1
                sensor_mask = torch.zeros_like(mask).float()
                sensor_mask[mask==2] = 1
                
                loss_ssim = ssim_loss(outputs, gts)
                loss_depth_smooth = depth_smoothness_loss(outputs, gts)
                loss_l1 = nn.L1Loss()(outputs, gts)
                

                depth_loss = 0.85*loss_ssim + loss_depth_smooth + 0.15*loss_l1 

                mask_loss = nn.CrossEntropyLoss()(mask_pred, masks)
                loss = depth_loss + mask_loss
                
                loss.backward()

                optimizer.step()
        
                running_loss += loss.item()
                
                images = denormalize(images)
                outputs = denormalize(outputs)
                gts = denormalize(gts)
                
                pbar.set_postfix(loss=loss.item(), mask_loss=mask_loss.item(), depth_loss=depth_loss.item())

                # Save the output of the i-th batch
                if (i % args.interval == 0):
                    gt_inter_mask = torch.zeros_like(masks).float()
                    gt_inter_mask[masks==1] = 1
                    gt_inter_mask = gt_inter_mask.unsqueeze(1)
                    gt_sensor_mask = torch.zeros_like(masks).float()
                    gt_sensor_mask[masks==2] = 1
                    gt_sensor_mask = gt_sensor_mask.unsqueeze(1)
                    for j in range(1):
                        save_depth(outputs[j].detach().cpu(), os.path.join(os.path.join(exp, output_dir), f'epoch_{epoch+1}_output_{i*args.batch_size+j:05d}.png'))

                        combined_outputs = torch.cat([minmax(images[j].detach().cpu()), minmax(outputs[j].detach().cpu()), minmax(gts[j].detach().cpu())], dim=2)
                        combined_masks = torch.cat([inter_mask[j].detach().cpu(), sensor_mask[j].detach().cpu(), gt_inter_mask[j].detach().cpu(), gt_sensor_mask[j].detach().cpu()], dim=2)
                        save_image(combined_outputs, os.path.join(os.path.join(exp, output_dir), f'epoch_{epoch+1}_combined_outputs_{i*args.batch_size+j:05d}.png'), normalize=False)
                        save_image(combined_masks, os.path.join(os.path.join(exp, output_dir), f'epoch_{epoch+1}_combined_masks_{i*args.batch_size+j:05d}.png'))
                break    
            scheduler.step()

            print(f"Epoch [{epoch+1}/{args.epoch}], Loss: {running_loss/len(train_dataloader)}")

        torch.save(model.state_dict(), os.path.join(os.path.join(exp, 'checkpoints'), f'latest.pth'))
        
        metrics = StreamSegMetrics(3)    
        val_rmse, val_psnr, val_score = evaluate(model, val_dataloader, device, args.batch_size, metrics, save_img=False, exp=exp)
        print(f'Val RMSE: {val_rmse:.6f}')
        print(f'Val PSNR: {val_psnr:.2f}')
        print()
        print(metrics.to_str(val_score))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            print(f'Best RMSE: {best_rmse:.6f}')
            torch.save(model.state_dict(), os.path.join(os.path.join(exp, 'checkpoints'), f'best.pth'))
        
    end = time.time()
    
    print('training time:', end-start, 's')
    print(f'Best Epoch: {best_epoch}')
    model.load_state_dict(torch.load(os.path.join(exp, 'checkpoints', f'best.pth')))
    
    print()
    test_rmse, test_psnr, test_score = evaluate(model, test_dataloader, device, args.batch_size, metrics, save_img=args.save_image, exp=exp)
    
    print(f'Test RMSE: {test_rmse:.6f}')
    print(f'Test PSNR: {test_psnr:.2f}')
    print()
    print(metrics.to_str(test_score))
