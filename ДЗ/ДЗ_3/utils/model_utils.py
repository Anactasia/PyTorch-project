import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # путь к корню проекта

from models import FullyConnectedModel
import torch
import os

def save_model(path, model, optimizer=None, epoch=None, extra_info=None):
    state = {
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if extra_info is not None:
        state.update(extra_info)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_model(path, model, optimizer=None):
    state = torch.load(path)
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    epoch = state.get('epoch', None)
    return epoch


def build_model_from_layers(layer_config, input_size=784, num_classes=10):
    config = {
        "input_size": input_size,
        "num_classes": num_classes,
        "layers": layer_config
    }
    return FullyConnectedModel(**config)

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
