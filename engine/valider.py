import torch
import os
import torch.nn as nn
import time
import cv2
from metrics.meter import AverageMeter, ProgressMeter
import data.img_proc as imgproc


def validation(config, g_model: nn.Module, data_prefetcher, epoch: int = 0,
               writer=None, psnr_model=None, ssim_model=None, mode="valid"):
    if mode == "valid":
        # Calculate how many batches of data are in each Epoch
        batch_time = AverageMeter("Time", ":6.3f")
        psnres = AverageMeter("PSNR", ":4.2f")
        ssimes = AverageMeter("SSIM", ":4.4f")
        progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")
    
        # convert mode to eval
        g_model.eval()
        batch_index = 0
        
        data_prefetcher.reset()
        batch_data = data_prefetcher.next()
        end = time.time()
        
        with torch.no_grad():
            while batch_data is not None:
                # tranfer data to GPU
                hr = batch_data['hr'].to(config['device'], non_blocking=True)
                lr = batch_data['lr'].to(config['device'], non_blocking=True)
                
                # generate super resolution image
                sr = g_model(lr)
                
                # compute psnr and ssim
                psnr = psnr_model(sr, hr)
                ssim = ssim_model(sr, hr)
                psnres.update(psnr.item(), lr.size(0))
                ssimes.update(ssim.item(), lr.size(0))
                
                # Calculate the time it takes to fully test a batch of data
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Record training log information
                if batch_index % config['train']['valid_print_frequency'] == 0:
                    progress.display(batch_index + 1)
                    
                batch_data = data_prefetcher.next()
                batch_index += 1
                
        # print metrics
        progress.display_summary()

        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)     

        return psnres.avg, ssimes.avg

    elif mode == "test":
        psnr_metrics = []
        ssim_metrics = []
        
        g_model.eval()
        batch_index = 0
        
        data_prefetcher.reset()
        batch_data = data_prefetcher.next()
        with torch.no_grad():
            while batch_data is not None:
                # tranfer data to GPU
                hr = batch_data['hr'].to(config['device'], non_blocking=True)
                lr = batch_data['lr'].to(config['device'], non_blocking=True)
                save_name = batch_data['name'][0]
                
                # generate super resolution image
                sr = g_model(lr)
                save_path = os.path.join(config['test_dir'], save_name)
                # Save image
                sr_image = imgproc.tensor_to_image(sr, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, sr_image)
                
                # compute psnr and ssim
                psnr = psnr_model(sr, hr)
                ssim = ssim_model(sr, hr)
                
                psnr_metrics.append(psnr.item())
                ssim_metrics.append(ssim.item())
                
                batch_data = data_prefetcher.next()
                batch_index += 1
                
                print(f"Validate image {save_name}: PSNR: {psnr.item():4.2f}, SSIM: {ssim.item():4.2f}")
            
        return psnr_metrics, ssim_metrics
            
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")
        
            
    