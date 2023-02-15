import argparse
import os
import sys
import cv2
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.generator import Generator
import data.img_proc as imgproc
from utils import load_state_dict, read_cfg, make_save_dir

def main(args):
    # Prepare model and device
    config = read_cfg('config/config.yaml')
    device = torch.device(args.device)
    g_model = Generator(scale_factor=config['dataset']['upscale_factor'], B=config['model']['num_rcb'],
                        channels=config['model']['channels'], in_channels=config['model']['in_channels'],
                        out_channels=config['model']['out_channels'])
    g_model = g_model.to(device)
    
    g_model = load_state_dict(model=g_model,
                              model_weight_path=args.g_model_path,
                              load_mode="pretrained")
    print("Load model from {} successfully".format(args.g_model_path))
    g_model.eval()
    
    # Prepare input data
    image_name = os.path.basename(args.input_path)
    lr_tensor = imgproc.preprocess_one_image(args.input_path, device)
    
    with torch.no_grad():
        sr_tensor = g_model(lr_tensor)
    
    # save image
    make_save_dir(config=config)
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(config['infer_dir'], image_name), sr_image)
    
    print("SR image saved to {}".format(os.path.join(config['infer_dir'], image_name)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SRGAN')
    parser.add_argument("--g_model_path", type=str, default="runs/train/g_model_best.pth",
                        help="path to g_model weights file")
    parser.add_argument("--input_path", type=str, default="data/test/valid_lr.png")
    parser.add_argument("--device", type=str, default="cpu", help="device to use for inference")
    args = parser.parse_args()
    
    main(args=args)
    