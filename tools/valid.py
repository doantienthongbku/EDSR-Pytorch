import os
import torch
import cv2
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import *
from metrics.quality_assesment import PSNR, SSIM
from engine.valider import validation


config = read_cfg('config/config.yaml')
checkpoint_path = config['test']['g_model_weights_path']

g_model, _ = build_model(config)
print("Build model successfully.")

checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
g_model = load_state_dict(model=g_model,
                          model_weight_path=checkpoint_path,
                          load_mode="pretrained")
print(f"Load checkpoint from {checkpoint_path} successfully.")

g_model.eval()

# define quality metrics
psnr_model = PSNR(config['dataset']['upscale_factor'], config['only_test_y_channel'])
ssim_model = SSIM(config['dataset']['upscale_factor'], config['only_test_y_channel'])
# transfer quality metrics to GPU
psnr_model = psnr_model.to(config['device'])
ssim_model = ssim_model.to(config['device'])

make_save_dir(config=config)

_, _, test_prefetcher = load_dataset(config=config)
psnr_list, ssim_list = validation(config, g_model, test_prefetcher, _, 
                        _, psnr_model, ssim_model, mode="test")

print("==========================SUMMARY==========================")
print("PSNR: {:.6f} dB".format(sum(psnr_list) / len(psnr_list)))
print("SSIM: {:.6f}".format(sum(ssim_list) / len(ssim_list)))

