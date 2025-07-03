import os
import time
import torch
from utils.model_utils import build_model_from_layers, count_parameters
from utils.experiment_utils import run_experiment, save_experiment_results
from utils.visualization_utils import plot_training_history
from datasets import get_mnist_loaders 


def generate_layers(depth, hidden_size=128, use_dropout=False, use_batchnorm=False):
    layers = []
    for _ in range(depth - 1):
        layers.append({"type": "linear", "size": hidden_size})
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if use_dropout:
            layers.append({"type": "dropout", "rate": 0.3})
    return layers


def run_depth_experiments(
    depth_values,
    dataset='mnist',
    epochs=5,
    lr=0.001,
    batch_size=64,
    device='cpu',
    use_dropout=False,
    use_batchnorm=False,
    save_important_only=True
):
    if dataset == 'mnist':
        input_size = 28 * 28
        num_classes = 10
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    else:
        raise NotImplementedError("Only MNIST supported for now")

    base_dir = os.path.dirname(os.path.abspath(__file__))  
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for depth in depth_values:
        print(f"\n Depth {depth} | Dropout: {use_dropout} | BatchNorm: {use_batchnorm}")

        layers = generate_layers(depth, use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        model = build_model_from_layers(layers, input_size=input_size, num_classes=num_classes).to(device)

        start = time.time()
        history = run_experiment(model, train_loader, test_loader, epochs, lr, device)
        duration = time.time() - start

        print(f"Training time: {duration:.2f} sec")
        print(f"Final Test Accuracy: {history['test_accs'][-1]:.4f}")
        print(f"Params: {count_parameters(model):,}")

        save_dir = os.path.join(base_dir, "results", "depth_experiments")
        experiment_name = f"depth_{depth}_dropout{use_dropout}_bn{use_batchnorm}"
        save_experiment_results(
            save_dir=save_dir,
            experiment_name=experiment_name,
            depth=depth,
            history=history,
            duration=duration,
            model=model,
            hidden_size=128,
            use_dropout=use_dropout,
            use_batchnorm=use_batchnorm,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
        )


        # Название графика
        name = f"{dataset}_depth{depth}_dropout{use_dropout}_bn{use_batchnorm}"
        plot_path = os.path.join(plot_dir, name + ".png")

        # Условие сохранения
        should_save = not save_important_only or (
            depth in [1, 3, 7] or
            use_dropout or
            use_batchnorm
        )

        if should_save:
            plot_training_history(history, save_path=plot_path)
        else:
            plot_training_history(history) 


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depths = [1, 2, 3, 5, 7]

    print("\n Без регуляризации:")
    run_depth_experiments(depths, device=device)

    print("\n С Dropout:")
    run_depth_experiments(depths, device=device, use_dropout=True)

    print("\n С BatchNorm:")
    run_depth_experiments(depths, device=device, use_batchnorm=True)

    print("\n С Dropout + BatchNorm:")
    run_depth_experiments(depths, device=device, use_dropout=True, use_batchnorm=True)
