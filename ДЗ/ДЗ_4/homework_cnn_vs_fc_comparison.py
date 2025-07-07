import torch
from models.fc_models import FCNet, DeepFCNet
from models.cnn_models import SimpleCNN, CNNWithResidual, CNNWithResidualReg
from datasets import get_mnist_loaders, get_cifar_loaders
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history, plot_confusion_matrix
from utils.comparison_utils import count_parameters, measure_inference_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 сравнение
train_loader, test_loader = get_cifar_loaders(batch_size=64)

cifar_models = {
    "DeepFCNet": DeepFCNet(input_size=3*32*32, num_classes=10),
    "CNNWithResidual": CNNWithResidual(input_channels=3, num_classes=10),
    "CNNWithResidualReg": CNNWithResidualReg(input_channels=3, num_classes=10),
}

cifar_histories = {}

for name, model in cifar_models.items():
    model = model.to(device)
    print(f"\nTraining {name} ({count_parameters(model)} parameters)")
    history = train_model(model, train_loader, test_loader, epochs=5, device=device)
    plot_training_history(history)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    inference_time = measure_inference_time(model, dummy_input, device=device)

    cifar_histories[name] = {
        "history": history,
        "params": count_parameters(model),
        "inference_time_ms": inference_time
    }

# Итоговая таблица
print("\nSummary for CIFAR-10:")
print(f"{'Model':<20} {'Params':<12} {'Inference(ms)':<15} {'Train Acc':<10} {'Test Acc':<10}")
for name, info in cifar_histories.items():
    h = info['history']
    print(f"{name:<20} {info['params']:<12} {info['inference_time_ms']:<15.2f} {h['train_accs'][-1]:<10.4f} {h['test_accs'][-1]:<10.4f}")

# Построение confusion matrix
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("\nConfusion Matrices (CIFAR-10):")
for name, model in cifar_models.items():
    print(f"\n{name}:")
    plot_confusion_matrix(model, test_loader, device, class_names)




#1.1 Задание
# # train_loader, test_loader = get_mnist_loaders(batch_size=64)
# train_loader, test_loader = get_cifar_loaders(batch_size=64)

# models = {
#     "FCNet": FCNet(),
#     "SimpleCNN": SimpleCNN(),
#     "CNNWithResidual": CNNWithResidual(),
# }

# histories = {}

# for name, model in models.items():
#     model = model.to(device)
#     print(f"Training {name} ({count_parameters(model)} parameters)")
#     history = train_model(model, train_loader, test_loader, epochs=5, device=device)
#     plot_training_history(history)

#     # Измеряем время инференса
#     dummy_input = torch.randn(1, 1, 28, 28).to(device)
#     inference_time = measure_inference_time(model, dummy_input, device=device)
#     print(f"Inference time for {name}: {inference_time:.2f} ms")

#     histories[name] = {
#         "history": history,
#         "params": count_parameters(model),
#         "inference_time_ms": inference_time
#     }

#     print("\nSummary:")
#     print(f"{'Model':<20} {'Params':<12} {'Inference(ms)':<15} {'Train Acc':<10} {'Test Acc':<10}")
#     for name, info in histories.items():
#         history = info['history']
#         params = info['params']
#         inference_time = info['inference_time_ms']
#         train_acc = history['train_accs'][-1]
#         test_acc = history['test_accs'][-1]
#         print(f"{name:<20} {params:<12} {inference_time:<15.2f} {train_acc:<10.4f} {test_acc:<10.4f}")
