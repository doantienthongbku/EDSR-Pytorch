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
train_prefetcher, valid_prefetcher, _ = load_dataset(config)
# build model
g_model, d_model = build_model(config)
# define criterion
l1_criterion, pixel_criterion, content_criterion, adversarial_criterion = define_loss(config)
# define optimizer
g_optimizer, d_optimizer = define_optimizer(config, g_model, d_model)
# define scheduler
g_scheduler, d_scheduler = define_scheduler(config, g_optimizer, d_optimizer)
print(" ... Done\n")

print("Check whether to load pretrained model weights...")
if config['train']['pretrained_g_model_weights_path']:
    g_model = load_state_dict(model=g_model,
                              model_weight_path=config['train']['pretrained_g_model_weights_path'],
                              load_mode="pretrained")
    print(f"Load pretrained g model weights from {config['train']['pretrained_g_model_weights_path']}")
else:
    print("No pretrained g model weights to load")

if config['train']['pretrained_d_model_weights_path']:
    d_model = load_state_dict(model=d_model,
                              model_weight_path=config['train']['pretrained_d_model_weights_path'],
                              load_mode="pretrained")
    print(f"Load pretrained d model weights from {config['train']['pretrained_d_model_weights_path']}")
else:
    print("No pretrained d model weights to load")
    
print("Check whether to continue training...")
if config['train']['resume']:
    g_model, g_optimizer, g_scheduler, start_epoch, best_psnr, best_ssim = \
        load_state_dict(model=g_model,
                        model_weight_path=config['train']['resume_g_model_weights_path'],
                        optimizer=g_optimizer,
                        scheduler=g_scheduler,
                        load_mode="resume")
    d_model, d_optimizer, d_scheduler, _, _, _ = \
        load_state_dict(model=d_model,
                        model_weight_path=config['train']['resume_d_model_weights_path'],
                        optimizer=d_optimizer,
                        scheduler=d_scheduler,
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
    train(config, g_model, d_model, train_prefetcher,
          pixel_criterion, content_criterion, adversarial_criterion,
          g_optimizer, d_optimizer, writer, epoch)
    psnr, ssim = validation(config, g_model, valid_prefetcher, epoch, 
                            writer, psnr_model, ssim_model, mode="valid")

    # Update LR
    d_scheduler.step()
    g_scheduler.step()
    
    is_best = psnr > best_psnr
    is_last = (epoch == config['train']['num_epochs'] - 1)
    best_psnr = max(psnr, best_psnr)
    best_ssim = max(ssim, best_ssim)
    
    # save model state dict
    g_state_dict = {"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": g_model.state_dict(),
                    "optimizer": g_optimizer.state_dict(),
                    "scheduler": g_scheduler.state_dict()}
    d_state_dict = {"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": d_model.state_dict(),
                    "optimizer": d_optimizer.state_dict(),
                    "scheduler": d_scheduler.state_dict()}    
        
    save_state_dict(state_dict=g_state_dict, save_name="g_model_epoch_{}.pth".format(epoch + 1), best_psnr=best_psnr,
                    save_dir=config['output_dir'], epoch=epoch, is_best=is_best, is_last=is_last, typ_model="g")
    save_state_dict(state_dict=d_state_dict, save_name="d_model_epoch_{}.pth".format(epoch + 1), best_psnr=best_psnr,
                    save_dir=config['output_dir'], epoch=epoch, is_best=is_best, is_last=is_last, typ_model="d")
    