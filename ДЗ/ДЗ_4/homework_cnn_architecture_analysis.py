import torch
from models.cnn_models import CNN3x3, CNN5x5, CNN7x7, CNNCombo1x1_3x3
from models.cnn_models import ShallowCNN, MediumCNN, DeepCNN, ResNetLikeCNN
from datasets import get_cifar_loaders
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history, plot_gradient_norms, visualize_feature_maps
from utils.comparison_utils import count_parameters, measure_inference_time

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем данные CIFAR-10
train_loader, test_loader = get_cifar_loaders(batch_size=64)


# # Задание 2.1: Размер ядра

# print("\n Kernel Size Analysis")
# kernel_models = {
#     "CNN3x3": CNN3x3(),
#     "CNN5x5": CNN5x5(),
#     "CNN7x7": CNN7x7(),
#     "CNN1x1+3x3": CNNCombo1x1_3x3(),
# }

# kernel_results = {}

# for name, model in kernel_models.items():
#     print(f"\n{name}:")
#     model = model.to(device)
#     print(f"Parameters: {count_parameters(model)}")
#     history = train_model(model, train_loader, test_loader, epochs=5, device=device)
#     plot_training_history(history)
#     dummy_input = torch.randn(1, 3, 32, 32).to(device)
#     inference_time = measure_inference_time(model, dummy_input, device=device)

#     kernel_results[name] = {
#         "history": history,
#         "params": count_parameters(model),
#         "inference_time": inference_time
#     }

# print("\nSummary: Kernel Size Analysis")
# print(f"{'Model':<15} {'Params':<10} {'Inference(ms)':<15} {'Train Acc':<10} {'Test Acc':<10}")
# for name, res in kernel_results.items():
#     h = res["history"]
#     print(f"{name:<15} {res['params']:<10} {res['inference_time']:<15.2f} {h['train_accs'][-1]:<10.4f} {h['test_accs'][-1]:<10.4f}")


# Задание 2.2: Глубина CNN

print("\nDepth Analysis")

depth_models = {
    "ShallowCNN": ShallowCNN(),
    "MediumCNN": MediumCNN(),
    "DeepCNN": DeepCNN(),
    "ResNetLikeCNN": ResNetLikeCNN(),
}

depth_results = {}

for name, model in depth_models.items():
    print(f"\n{name}:")
    model = model.to(device)
    print(f"Parameters: {count_parameters(model)}")
    
    history = train_model(model, train_loader, test_loader, epochs=3, device=device)
    plot_training_history(history)
    plot_gradient_norms(history["grad_norms"])
    
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    inference_time = measure_inference_time(model, dummy_input, device=device)

    # Визуализация feature maps первого слоя для примера
    sample_batch, _ = next(iter(test_loader))
    visualize_feature_maps(model, sample_batch[0], device)

    depth_results[name] = {
        "history": history,
        "params": count_parameters(model),
        "inference_time": inference_time
    }

    print("\nSummary: Depth Analysis")
    print(f"{'Model':<15} {'Params':<10} {'Inference(ms)':<15} {'Train Acc':<10} {'Test Acc':<10}")
    for name, res in depth_results.items():
        h = res["history"]
        print(f"{name:<15} {res['params']:<10} {res['inference_time']:<15.2f} {h['train_accs'][-1]:<10.4f} {h['test_accs'][-1]:<10.4f}")