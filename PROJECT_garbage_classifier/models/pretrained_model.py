from torchvision import models
import torch.nn as nn

def get_pretrained_model(model_name='resnet18', num_classes=2):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Модель не поддерживается")
    
    return model
