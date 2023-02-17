import yaml
import torch
import os
import shutil
import ast
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data.dataset import TrainValidImageDataset, TestImageDataset,\
    PrefetchGenerator, PrefetchDataLoader, CPUPrefetcher, CUDAPrefetcher
from model.EDSR import EDSR
from metrics.losses import ContentLoss

def read_cfg(cfg_file):
    """Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg


def load_dataset(config, mode='train'):
    if mode == 'train':
        # load train and valid dataset
        train_dataset = TrainValidImageDataset(
            images_dir=config['dataset']['train_hr_images_dir'],
            crop_size=config['dataset']['crop_size'],
            upscale_factor=config['dataset']['upscale_factor'],
            mode='train',
            image_format=config['dataset']['image_format']
        )
        
        valid_dataset = TrainValidImageDataset(
            images_dir=config['dataset']['valid_hr_images_dir'],
            crop_size=config['dataset']['crop_size'],
            upscale_factor=config['dataset']['upscale_factor'],
            mode='valid',
            image_format=config['dataset']['image_format']
        )
        
            # Generate dataloader
        train_loader = DataLoader(
            train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True,
            num_workers=config['dataset']['num_workers'], pin_memory=True, drop_last=True, persistent_workers=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=1, shuffle=False,
            num_workers=config['dataset']['num_workers'], pin_memory=True, drop_last=False, persistent_workers=True
        )
        
        train_prefetcher = CUDAPrefetcher(train_loader, config['device'])
        valid_prefetcher = CUDAPrefetcher(valid_loader, config['device'])
        
        return train_prefetcher, valid_prefetcher
    elif mode == 'test':
        test_dataset = TestImageDataset(
            test_hr_image_dir=config['dataset']['test_hr_images_dir'],
            test_lr_image_dir=config['dataset']['test_lr_images_dir'],
            image_format=config['dataset']['image_format']
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1,
            pin_memory=True, drop_last=False, persistent_workers=True
        )
        
        test_prefetcher = CUDAPrefetcher(test_loader, config['device'])
        
        return test_prefetcher
    else:
        raise ValueError("mode must be 'train' or 'test'")


def build_model(config):
    model = EDSR(scale_factor=config['dataset']['upscale_factor'],
                 B=config['model']['num_rcb'], channels=config['model']['channels'],
                 in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'],
                 rgb_range=config['model']['rgb_range'])

    if 'cuda' in config['device']:
        model = model.to(config['device'])
        
    return model

def define_loss(config):
    pixel_loss = nn.MSELoss()
    content_loss = ContentLoss(
        feature_model_extractor_node=config['train']["feature_model_extractor_node"],
        feature_model_normalize_mean=config['train']["feature_model_normalize_mean"],
        feature_model_normalize_std=config['train']["feature_model_normalize_std"]
    )
    l1_loss = nn.L1Loss()
    
    # Transfer to device
    if 'cuda' in config['device']:
        pixel_loss = pixel_loss.to(config["device"])
        content_loss = content_loss.to(config["device"])
        l1_loss = l1_loss.to(config["device"])
    
    return l1_loss, pixel_loss, content_loss


def define_optimizer(config, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=float(config['train']['model_lr']),
                                betas=ast.literal_eval(config['train']['model_betas']),
                                eps=float(config['train']['model_eps']),
                                weight_decay=float(config['train']['model_weight_decay']))
    
    return optimizer


def define_scheduler(config, optimizer: torch.optim.Adam):
    scheduler = lr_scheduler.StepLR(optimizer,
                                    config['train']['lr_scheduler_step_size'],
                                    config['train']['lr_scheduler_gamma'])
    return scheduler


def load_state_dict(model: nn.Module, model_weight_path: str, 
                    optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None,
                    load_mode: str = "pretrained"):
    checkpoint = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    
    if load_mode == "pretrained":
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint['state_dict'].items()
                      if k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        
        return model
    elif load_mode == "resume":
        restart_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint['state_dict'].items()
                      if k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        model_state_dict.update(state_dict)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        return model, optimizer, scheduler, restart_epoch, best_psnr, best_ssim
    else:
        raise ValueError("load_mode must be one of ['pretrained', 'resume']")


def save_state_dict(state_dict: dict, save_name: str, save_dir: str, best_psnr: float,
                    epoch: int, is_best: bool = False, is_last: bool = False):
    checkpoint_path = os.path.join(save_dir, save_name)
    torch.save(state_dict, checkpoint_path)
    
    if (is_best):
        shutil.copyfile(checkpoint_path, os.path.join(save_dir, 'model_best.pth'))
        print("save best model at epoch {} with psnr {}".format(epoch + 1, best_psnr))
    if (is_last):
        shutil.copyfile(checkpoint_path, os.path.join(save_dir, 'model_last.pth'))


def make_save_dir(config):
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    if not os.path.exists(config['test_dir']):
        os.makedirs(config['test_dir'])
    if not os.path.exists(config['infer_dir']):
        os.makedirs(config['infer_dir'])
        
    print("Save directory is created.")
