import os
import queue
import threading

import glob
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import data.img_proc as imgproc

# this code build base on https://github.com/Lornatang/SRGAN-PyTorch/blob/main/dataset.py

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

class TrainValidImageDataset(Dataset):
    
    def __init__(self, images_dir: str = "./datasets", crop_size: int = 96,
                 upscale_factor: int = 2, mode: str = "train", image_format: str = "png") -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_path_list = glob.glob(images_dir + "/*." + image_format)
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __len__(self) -> int:
        return len(self.image_path_list)
    
    def __getitem__(self, index: int):
        # Read a batch of image data
        image = cv2.imread(self.image_path_list[index]).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "train":
            hr_crop_image = imgproc.random_crop(image, self.crop_size)
        elif self.mode == "valid":
            hr_crop_image = imgproc.center_crop(image, self.crop_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(hr_crop_image, 1 / self.upscale_factor)

        # BGR convert RGB
        hr_crop_image = cv2.cvtColor(hr_crop_image, cv2.COLOR_BGR2RGB)
        lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        hr_crop_tensor = imgproc.image_to_tensor(hr_crop_image, False, False)
        lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)

        return {"lr": lr_crop_tensor, "hr": hr_crop_tensor}
    

class TestImageDataset(Dataset):
    
    def __init__(self, test_hr_image_dir: str, test_lr_image_dir: str, image_format: str = "png") -> None:
        super(TestImageDataset, self).__init__()
        self.test_hr_image_path_list = glob.glob(test_hr_image_dir + "/*." + image_format)
        self.test_lr_image_path_list = glob.glob(test_lr_image_dir + "/*." + image_format)
        
    def __len__(self) -> int:
        return len(self.test_hr_image_path_list)
    
    def __getitem__(self, index):
        # get image name
        image_name = os.path.basename(self.test_hr_image_path_list[index])
        
        # Get image from source path
        hr_image = cv2.imread(self.test_hr_image_path_list[index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.test_lr_image_path_list[index]).astype(np.float32) / 255.
        
        # BGR convert RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        hr_image_tensor = imgproc.image_to_tensor(hr_image, False, False)
        lr_image_tensor = imgproc.image_to_tensor(lr_image, False, False)
        
        return {"lr": lr_image_tensor, "hr": hr_image_tensor, "name": image_name}
    
    
class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.
    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.
    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
        