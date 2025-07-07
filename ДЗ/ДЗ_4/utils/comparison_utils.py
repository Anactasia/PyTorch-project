import torch
import matplotlib.pyplot as plt
import time




def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model

def measure_inference_time(model, input_tensor, device='cpu', runs=1000):
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(input_tensor)
        end = time.perf_counter()
    return ((end - start) / runs) * 1000

