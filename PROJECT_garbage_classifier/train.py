import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import get_dataloaders
from models.cnn import SimpleCNN
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  

def get_class_weights(train_dir, class_names):
    class_counts = []
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        count = len(os.listdir(class_path))
        class_counts.append(count)
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    
    # Инвертируем количество и нормируем
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.mean()
    return class_weights

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def train_model(data_dir, epochs=20, batch_size=32, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)

    num_classes = len(class_names)
    
    model = SimpleCNN(num_classes=num_classes).to(device)
    save_path = 'best_model3.pth'
        # Попытка загрузить сохранённую модель
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))



    train_loader, val_loader, test_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    train_dir = os.path.join(data_dir, 'train')
    class_weights = get_class_weights(train_dir, class_names)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    save_path = 'best_model3.pth'

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Добавляем tqdm для train
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Добавляем tqdm для val
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch} [Val]', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch}/{epochs} — '
              f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | '
              f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Лучшее состояние модели сохранено с точностью: {best_val_acc:.4f}')

    print('Обучение завершено.')



    # Построим графики
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))

    smooth_train_losses = smooth_curve(train_losses)
    smooth_val_losses = smooth_curve(val_losses)
    smooth_train_acc = smooth_curve(train_accuracies)
    smooth_val_acc = smooth_curve(val_accuracies)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, smooth_train_losses, label='Train Loss')
    plt.plot(epochs_range, smooth_val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs (Smoothed)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, smooth_train_acc, label='Train Acc')
    plt.plot(epochs_range, smooth_val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs (Smoothed)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_dir = 'PROJECT_garbage_classifier/data'
    train_model(data_dir)
