import os
import time
import torch
from utils.model_utils import build_model_from_layers, count_parameters
from utils.experiment_utils import run_experiment, save_experiment_results
from utils.visualization_utils import plot_training_history, plot_weight_distribution, plot_weight_distributions_comparison
from datasets import get_mnist_loaders


def build_regularized_model(dropout_rate=0.0, use_batchnorm=False, input_size=784, num_classes=10, hidden_size=128):
    layers = []
    for _ in range(2):  # 2 скрытых слоя
        layers.append({"type": "linear", "size": hidden_size})
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if dropout_rate > 0.0:
            layers.append({"type": "dropout", "rate": dropout_rate})
    return build_model_from_layers(layers, input_size=input_size, num_classes=num_classes)


def run_regularization_experiment(
    label,
    dropout_rate=0.0,
    use_batchnorm=False,
    weight_decay=0.0,
    device='cpu',
    save_dir="results/regularization_experiments",
    plot_dir="plots",
    epochs=3,
    lr=0.001,
    batch_size=64
):
    input_size = 28 * 28
    num_classes = 10
    train_loader, test_loader = get_mnist_loaders(batch_size)

    model = build_regularized_model(
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        input_size=input_size,
        num_classes=num_classes
    ).to(device)

    start = time.time()
    history = run_experiment(model, train_loader, test_loader, epochs, lr, device, weight_decay=weight_decay)
    duration = time.time() - start

    print(f"\n {label}")
    print(f"Test Accuracy: {history['test_accs'][-1]:.4f}")
    print(f"Training Time: {duration:.2f}s")
    print(f"Params: {count_parameters(model):,}")

    experiment_name = f"reg_{label.replace(' ', '_')}"
    save_experiment_results(
        save_dir=save_dir,
        experiment_name=experiment_name,
        history=history,
        duration=duration,
        model=model,
        use_dropout=dropout_rate > 0,
        use_batchnorm=use_batchnorm,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    os.makedirs(plot_dir, exist_ok=True)
    plot_training_history(history, save_path=os.path.join(plot_dir, f"{experiment_name}_history.png"))
    plot_weight_distribution(model, save_path=os.path.join(plot_dir, f"{experiment_name}_weights.png"))


def build_adaptive_regularized_model(
    dropout_rates,  # список dropout для каждого слоя (адаптивные коэффициенты)
    batchnorm_momentums,  # список momentum для batchnorm для каждого слоя (адаптивные параметры)
    input_size=784,
    num_classes=10,
    hidden_sizes=[128, 128]
):
    layers = []
    for i, hidden_size in enumerate(hidden_sizes):
        layers.append({"type": "linear", "size": hidden_size})

        # Добавляем BatchNorm с заданным momentum для каждого слоя (адаптивная нормализация)
        if batchnorm_momentums and batchnorm_momentums[i] is not None:
            layers.append({"type": "batch_norm", "momentum": batchnorm_momentums[i]})

        layers.append({"type": "relu"})

        # Добавляем Dropout с изменяющимся коэффициентом для каждого слоя (адаптивная регуляризация)
        if dropout_rates and dropout_rates[i] > 0.0:
            layers.append({"type": "dropout", "rate": dropout_rates[i]})

    return build_model_from_layers(layers, input_size=input_size, num_classes=num_classes)


def run_adaptive_regularization_experiment(
    label,
    dropout_schedule,  # функция, которая возвращает список dropout-коэффициентов для каждого слоя на текущей эпохе
    batchnorm_momentums,  # фиксированные или адаптивные значения momentum для BatchNorm по слоям
    device='cpu',
    save_dir="results/adaptive_regularization",
    plot_dir="plots",
    epochs=10,
    lr=0.001,
    batch_size=64
):
    input_size = 28 * 28
    num_classes = 10
    train_loader, test_loader = get_mnist_loaders(batch_size)

    hidden_sizes = [128, 128]
    all_histories = []

    for epoch in range(epochs):
        # Получаем текущие коэффициенты Dropout для каждого слоя по расписанию
        current_dropout_rates = dropout_schedule(epoch)

        # Строим модель с текущими адаптивными параметрами регуляризации
        model = build_adaptive_regularized_model(
            dropout_rates=current_dropout_rates,
            batchnorm_momentums=batchnorm_momentums,
            input_size=input_size,
            num_classes=num_classes,
            hidden_sizes=hidden_sizes,
        ).to(device)

        # Запускаем эксперимент на 1 эпоху, чтобы динамически менять регуляризацию
        start = time.time()
        history = run_experiment(model, train_loader, test_loader, 1, lr, device)
        duration = time.time() - start

        print(f"Epoch {epoch+1}/{epochs} | Test Acc: {history['test_accs'][-1]:.4f} | Time: {duration:.2f}s")
        all_histories.append(history)

    experiment_name = label.replace(" ", "_")
    save_experiment_results(
        save_dir=save_dir,
        experiment_name=experiment_name,
        history=all_histories[-1],
        duration=sum(h.get('duration', 0) for h in all_histories),
        model=model,
        use_dropout=True,
        use_batchnorm=True,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    os.makedirs(plot_dir, exist_ok=True)
    plot_training_history(all_histories[-1], save_path=os.path.join(plot_dir, f"{experiment_name}_history.png"))
    plot_weight_distribution(model, save_path=os.path.join(plot_dir, f"{experiment_name}_weights.png"))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # --- 3.1 Запуск фиксированных регуляризаций ---
    # regularization_setups = [
    #     ("No regularization", 0.0, False, 0.0),
    #     ("Dropout 0.1", 0.1, False, 0.0),
    #     ("Dropout 0.3", 0.3, False, 0.0),
    #     ("Dropout 0.5", 0.5, False, 0.0),
    #     ("BatchNorm only", 0.0, True, 0.0),
    #     ("Dropout 0.3 + BatchNorm", 0.3, True, 0.0),
    #     ("L2 Regularization (wd=0.01)", 0.0, False, 0.01),
    # ]

    # for label, dropout_rate, use_bn, weight_decay in regularization_setups:
    #     run_regularization_experiment(
    #         label=label,
    #         dropout_rate=dropout_rate,
    #         use_batchnorm=use_bn,
    #         weight_decay=weight_decay,
    #         device=device,
    #         epochs=3,
    #         lr=0.001,
    #         batch_size=64
    #     )

    # 3.2 Запуск адаптивной регуляризации 
    models = []
    labels = []
    # Функция, задающая динамическое изменение dropout-коэффициента по эпохам
    # Здесь dropout растёт линейно от 0 до 0.5 за 10 эпох
    def dropout_schedule(epoch):
        max_epochs = 10
        rate = 0.5 * (epoch / max_epochs)
        return [rate, rate]  # применяем одинаковый коэффициент к двум слоям

    # Задаём разные значения momentum для BatchNorm по слоям
    batchnorm_momentums = [0.1, 0.5]

    model_adaptive = run_adaptive_regularization_experiment(
        label="Adaptive Dropout + BatchNorm momentum",
        dropout_schedule=dropout_schedule,
        batchnorm_momentums=batchnorm_momentums,
        device=device,
        epochs=10,
        lr=0.001,
        batch_size=64
    )

    models.append(model_adaptive)
    labels.append("Adaptive Dropout + BatchNorm momentum")


    # Визуализируем распределения весов по слоям для всех моделей
    plot_weight_distributions_comparison(models, labels, save_dir="plots", experiment_name="regularization_comparison")
