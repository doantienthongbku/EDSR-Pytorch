import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import read_cfg, load_dataset, build_model, define_optimizer, \
    define_scheduler, define_loss, load_state_dict, save_state_dict, \
    make_save_dir
from metrics.quality_assesment import PSNR, SSIM
from engine.trainer import train
from engine.valider import validation


# Initialize the number of training epochs
start_epoch = 0
# Initialize training to generate network evaluation indicators
best_psnr = 0.0
best_ssim = 0.0

# read config file
config = read_cfg('config/config.yaml')
# turn on cudnn if True
if config['cudnn_benchmark']: cudnn.benchmark = True

print("Building dataset, model, optimizer, criterion, and scheduler ...")
# load datasets
train_prefetcher, valid_prefetcher = load_dataset(config, mode="train")
# build model
model = build_model(config)
# define criterion
l1_criterion, pixel_criterion, content_criterion = define_loss(config)
# define optimizer
optimizer = define_optimizer(config, model)
# define scheduler
scheduler = define_scheduler(config, optimizer)
print("Dataset, model, optimizer, criterion, and scheduler build successfully!\n")

print("Check whether to load pretrained model weights...")
if config['train']['pretrained_model_weights_path']:
    model = load_state_dict(model=model,
                            model_weight_path=config['train']['pretrained_model_weights_path'],
                            load_mode="pretrained")
    print(f"Load pretrained model weights from {config['train']['pretrained_model_weights_path']}")
else:
    print("No pretrained model weights to load")
    
print("Check whether to continue training...")
if config['train']['resume']:
    model, optimizer, scheduler, start_epoch, best_psnr, best_ssim = \
        load_state_dict(model=model,
                        model_weight_path=config['train']['resume_model_weights_path'],
                        optimizer=optimizer,
                        scheduler=scheduler,
                        load_mode="resume")
    print(f"Continue training from epoch {start_epoch}")
    
# make save dir
make_save_dir(config)

# define tensorboard writer
writer = SummaryWriter(log_dir=config['log_dir'])

# define quality metrics
psnr_model = PSNR(config['dataset']['upscale_factor'], config['only_test_y_channel'])
ssim_model = SSIM(config['dataset']['upscale_factor'], config['only_test_y_channel'])
# transfer quality metrics to GPU
psnr_model = psnr_model.to(config['device'])
ssim_model = ssim_model.to(config['device'])

for epoch in range(start_epoch, config['train']['num_epochs']):
    train(epoch, config, model, train_prefetcher,
          l1_criterion, pixel_criterion, content_criterion,
          optimizer, writer)
    psnr, ssim = validation(config, model, valid_prefetcher, epoch, 
                            writer, psnr_model, ssim_model, mode="valid")

    # Update LR
    scheduler.step()
    
    is_best = psnr > best_psnr
    is_last = (epoch == config['train']['num_epochs'] - 1)
    best_psnr = max(psnr, best_psnr)
    best_ssim = max(ssim, best_ssim)
    
    # save model state dict
    state_dict = {"epoch": epoch + 1,
                  "best_psnr": best_psnr,
                  "best_ssim": best_ssim,
                  "state_dict": model.state_dict(),
                  "optimizer": optimizer.state_dict,
                  "scheduler": scheduler.state_dict()}
        
    save_state_dict(state_dict=state_dict, save_name=f"{config['model']['arch_name']}_epoch_{epoch+1}.pth", best_psnr=best_psnr,
                    save_dir=config['output_dir'], epoch=epoch, is_best=is_best, is_last=is_last)
    