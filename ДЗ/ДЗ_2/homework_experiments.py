
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from homework_datasets import CustomDataset
from homework_model_modification import LinearRegressionModel
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_model(model, filename):
    os.makedirs('models', exist_ok=True)
    path = os.path.join('models', filename)
    torch.save(model.state_dict(), path)
    logging.info(f"Модель сохранена: {path}")



def run_experiment(X, y, optimizer_name='SGD', lr=0.01, batch_size=32, epochs=50, save_name=None):
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = LinearRegressionModel(X.shape[1])
    criterion = nn.MSELoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr) 
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) 
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        raise ValueError("Unknown optimizer")

    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_X, val_y in val_loader:
                val_pred = model(val_X)
                val_loss += criterion(val_pred, val_y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

    # Восстанавливаем лучшую модель по валидации
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Сохраняем модель, если указано имя файла
    if save_name is not None:
        save_model(model, save_name)

    return model, val_losses



def plot_results(results, title, filename):
    

    plt.figure(figsize=(10, 6))
    for label, losses in results.items():

        plt.plot(losses, label=label)
    plt.xlabel("Эпоха")                     
    plt.ylabel("Ошибка на валидации")       
    plt.title(title.replace("Comparison", "Сравнение"))  
    plt.legend(title="Параметры")          
    plt.grid(True)

  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)

    plt.savefig(save_path)
    plt.close()
    logging.info(f"Сохранён график: {save_path}")




def feature_engineering(X):
    # Создание новых признаков
    X_np = X.numpy()
    poly_features = X_np ** 2
    interaction = X_np[:, 0:1] * X_np[:, 1:2]
    mean_feature = np.mean(X_np, axis=1, keepdims=True)
    std_feature = np.std(X_np, axis=1, keepdims=True)
    X_new = np.hstack([X_np, poly_features, interaction, mean_feature, std_feature])
    return torch.tensor(X_new, dtype=torch.float32)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'insurance.csv')

    ds = CustomDataset(data_path)
    X, y = ds.get_features_and_targets_regression()

    # 1. Гиперпараметры
    logging.info("Исследование learning rate")
    results_lr = {}
    for lr in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        label = f"LR={lr}"
        model, losses = run_experiment(X, y, optimizer_name='SGD', lr=lr)
        results_lr[label] = losses
    plot_results(results_lr, "Сравнение скоростей обучения", "lr_comparison.png")

    logging.info("Исследование оптимизаторов")
    results_opt = {}
    for opt in ['SGD', 'Adam', 'RMSprop']:
        model, losses = run_experiment(X, y, optimizer_name=opt, lr=0.01)
        results_opt[opt] = losses
    plot_results(results_opt, "Сравнение оптимизаторов",   "optimizer_comparison.png")

    logging.info("Исследование batch size")
    results_bs = {}
    for bs in [4, 8, 16, 32, 64, 128]:
        label = f"BS={bs}"
        model, losses = run_experiment(X, y, optimizer_name='SGD', lr=0.01, batch_size=bs)
        results_bs[label] = losses
    plot_results(results_bs,  "Сравнение размеров батча",  "batchsize_comparison.png")


    # 2. Feature Engineering
    logging.info("Feature Engineering")
    X_fe = feature_engineering(X)
    base_loss = run_experiment(X, y, optimizer_name='SGD', lr=0.01)[-1]
    fe_loss = run_experiment(X_fe, y, optimizer_name='SGD', lr=0.01)[-1]
    print(f"Baseline loss: {base_loss[0]:.2f}")
    print(f"Feature engineered loss: {fe_loss[0]:.2f}")


if __name__ == "__main__":
    main()
