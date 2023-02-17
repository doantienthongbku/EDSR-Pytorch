import torch
import torch.nn as nn
import time
from metrics.meter import AverageMeter, ProgressMeter


def train(epoch, config, model: nn.Module, train_prefetcher,
          l1_criterion, pixel_criterion, content_criterion,
          optimizer, writer) -> None:
    num_batches = len(train_prefetcher)
    # print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    l1_losses = AverageMeter("L1_Loss:", ":6.6f")
    pixel_losses = AverageMeter("Pixel_L:", ":6.6f")
    content_losses = AverageMeter("Content_L", ":6.6f")
    
    progress = ProgressMeter(num_batches,
                            [batch_time, l1_losses, pixel_losses, content_losses],
                            prefix=f"Epoch: [{epoch + 1}/{config['train']['num_epochs']}]")
    
    # convert mode to train
    model.train()
    
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    
    # Calculate the initial time 
    end = time.time()
    batch_index = 0
    
    while batch_data is not None:
        # Transfer data to GPU
        hr = batch_data['hr'].to(config['device'], non_blocking=True)
        lr = batch_data['lr'].to(config['device'], non_blocking=True)
        
        # Initialize model gradients and compute sr image
        model.zero_grad(set_to_none=True)
        sr = model(lr)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        l1_loss = config['train']['l1_weight'] * l1_criterion(sr, hr)
        pixel_loss = config['train']['pixel_weight'] * pixel_criterion(sr, hr)
        content_loss = config['train']['content_weight'] * content_criterion(sr, hr)
        
        loss = l1_loss + pixel_loss + content_loss
        loss.backward()
        optimizer.step()
        
        # Statistical accuracy and loss value for terminal data output
        l1_losses.update(l1_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Write the data during training to the training log file
        if batch_index % config['train']['train_print_frequency'] == 0:
            iters = batch_index + epoch * num_batches + 1
            writer.add_scalar("Train/Loss", loss.item(), iters)
            writer.add_scalar("Train/L1_Loss", l1_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            progress.display(batch_index + 1)
            
        batch_data = train_prefetcher.next()
        batch_index += 1