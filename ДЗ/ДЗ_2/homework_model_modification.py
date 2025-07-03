import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os




def precision_recall_f1_multiclass(y_true, y_pred, num_classes):
    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


# Модели

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

class MultiClassLogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

# Функции для сохранения

def save_model(model, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Папка с этим скриптом
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), path)




def train_linreg(X, y, l1_lambda=0.0, l2_lambda=0.0, patience=5, epochs=100, batch_size=32, lr=0.01):
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LinearRegressionModel(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm
            if l2_lambda > 0:
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss += l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_pred = model(val_X)
                loss = criterion(val_pred, val_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Эпоха  {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            # Сохраняем лучшую модель
            save_model(model, 'best_linear_regression.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def train_logreg(X, y, num_classes, epochs=50, batch_size=32, lr=0.1):
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Подсчёт классов только на обучающей выборке
    train_labels = torch.stack([label for _, label in train_dataset])

    counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes  # нормализация

    class_weights = weights.to(X.device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MultiClassLogisticRegression(X.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                logits = model(val_X)
                loss = criterion(logits, val_y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(val_y)

        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        precision, recall, f1 = precision_recall_f1_multiclass(all_labels, all_preds, num_classes)

        print(f"Эпоха  {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            save_model(model, 'best_multiclass_logistic_regression.pth')


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model



#  Пример данных и запуск 

if __name__ == '__main__':
    # Генерация данных для линейной регрессии
    n_samples = 500
    n_features = 6

    X_lin = torch.randn(n_samples, n_features)
    true_weights = torch.randn(n_features, 1)
    y_lin = X_lin @ true_weights + 0.1 * torch.randn(n_samples, 1)

    train_linreg(X_lin, y_lin, l1_lambda=0.001, l2_lambda=0.001, patience=7, epochs=100)

    # Генерация данных для многоклассовой логистической регрессии
    num_classes = 3
    X_log = torch.randn(n_samples, n_features)
    y_log = torch.randint(0, num_classes, (n_samples,))

    train_logreg(X_log, y_log, num_classes=num_classes, epochs=50)