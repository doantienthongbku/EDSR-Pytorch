import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


class ContentLoss(nn.Module):
    def __init__(self,
                 feature_model_extractor_node: str = "features.35",
                 feature_model_normalize_mean: list = [0.485, 0.456, 0.406],
                 feature_model_normalize_std: list = [0.229, 0.224, 0.225]):
        super(ContentLoss, self).__init__()
        self.feature_model_extractor_node = feature_model_extractor_node
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = create_feature_extractor(model, [self.feature_model_extractor_node])
        self.normalize = transforms.Normalize(mean=feature_model_normalize_mean, std=feature_model_normalize_std)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, sr_image, hr_image):
        sr_image = self.normalize(sr_image)
        hr_image = self.normalize(hr_image)
        sr_image_feature = self.feature_extractor(sr_image)[self.feature_model_extractor_node]
        hr_image_feature = self.feature_extractor(hr_image)[self.feature_model_extractor_node]
        loss = F.mse_loss(sr_image_feature, hr_image_feature)
        
        return loss
