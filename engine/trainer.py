import torch
import torch.nn as nn
import time
from metrics.meter import AverageMeter, ProgressMeter


def train(config, g_model: nn.Module, d_model: nn.Module, train_prefetcher,
          pixel_criterion, content_criterion, adversarial_criterion,
          g_optimizer, d_optimizer, writer, epoch) -> None:
    num_batches = len(train_prefetcher)
    # print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    pixel_losses = AverageMeter("Pixel_L:", ":6.6f")
    content_losses = AverageMeter("Content_L", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial_L", ":6.6f")
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    
    progress = ProgressMeter(num_batches,
                            [batch_time, pixel_losses, content_losses, adversarial_losses,
                             d_gt_probabilities, d_sr_probabilities],
                            prefix=f"Epoch: [{epoch + 1}/{config['train']['num_epochs']}]")
    
    # convert mode to train
    g_model.train()
    d_model.train()
    
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    
    # Calculate the initial time 
    end = time.time()
    batch_index = 0
    
    while batch_data is not None:
        # Transfer data to GPU
        hr = batch_data['hr'].to(config['device'], non_blocking=True)
        lr = batch_data['lr'].to(config['device'], non_blocking=True)
        
        # set label for discriminator
        batch_size, _, heigth, width = hr.shape
        hr_label = torch.full([batch_size], 1.0, dtype=hr.dtype, device=config['device'])        
        lr_label = torch.full([batch_size], 0.0, dtype=hr.dtype, device=config['device'])
        
        # Enable gradient calculation of discriminator
        for d_params in d_model.parameters():
            d_params.requires_grad = True
        # Initialize discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        hr_output = d_model(hr)
        d_loss_hr = adversarial_criterion(hr_output, hr_label)
        # backward propagation
        d_loss_hr.backward(retain_graph=True)
        
        # calculate the output of the discriminator model for fake samples
        sr = g_model(lr)
        sr_output = d_model(sr.detach().clone())
        d_loss_sr = adversarial_criterion(sr_output, lr_label)
        # backward propagation
        d_loss_sr.backward()
        
        d_loss = d_loss_hr + d_loss_sr
        d_optimizer.step()
        
        # Disable gradient calculation of discriminator
        for d_params in d_model.parameters():
            d_params.requires_grad = False
            
        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)
        
        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        pixel_loss = config['train']['pixel_weight'] * pixel_criterion(sr, hr)
        content_loss = config['train']['content_weight'] * content_criterion(sr, hr)
        adversarial_loss = config['train']['adversarial_weight'] * adversarial_criterion(d_model(sr), hr_label)
        
        g_loss = pixel_loss + content_loss + adversarial_loss
        g_loss.backward()
        g_optimizer.step()
        
        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_gt_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))
        
        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_gt_probabilities.update(d_gt_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Write the data during training to the training log file
        if batch_index % config['train']['train_print_frequency'] == 0:
            iters = batch_index + epoch * num_batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index + 1)
            
        batch_data = train_prefetcher.next()
        batch_index += 1