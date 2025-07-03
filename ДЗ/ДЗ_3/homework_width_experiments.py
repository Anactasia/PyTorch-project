import os
import time
import torch
from utils.model_utils import build_model_from_layers, count_parameters
from utils.experiment_utils import run_experiment, save_experiment_results
from utils.visualization_utils import plot_heatmap
from datasets import get_mnist_loaders


def generate_width_layers(sizes, use_dropout=False, use_batchnorm=False):
    layers = []
    for size in sizes:
        layers.append({"type": "linear", "size": size})
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if use_dropout:
            layers.append({"type": "dropout", "rate": 0.3})
    return layers


def run_width_experiment(sizes, dataset, epochs, lr, batch_size, device, use_dropout, use_batchnorm):
    if dataset == 'mnist':
        input_size = 28 * 28
        num_classes = 10
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    else:
        raise NotImplementedError("Only MNIST supported")

    layers = generate_width_layers(sizes, use_dropout=use_dropout, use_batchnorm=use_batchnorm)
    model = build_model_from_layers(layers, input_size=input_size, num_classes=num_classes).to(device)

    start = time.time()
    history = run_experiment(model, train_loader, test_loader, epochs, lr, device)
    duration = time.time() - start

    # Сохраняем результаты
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "results", "width_experiments")
    experiment_name = f"widths_{'_'.join(map(str, sizes))}_dropout{use_dropout}_bn{use_batchnorm}"
    save_experiment_results(
        save_dir=save_dir,
        experiment_name=experiment_name,
        widths=sizes,
        history=history,
        duration=duration,
        model=model,
        use_dropout=use_dropout,
        use_batchnorm=use_batchnorm,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    return history['test_accs'][-1], duration, count_parameters(model)


def run_basic_width_experiments(width_configs, device):
    print("### 2.1 Basic Width Experiments ###")
    for sizes in width_configs:
        print(f"\n Widths: {sizes}")
        acc, duration, params = run_width_experiment(
            sizes, dataset='mnist', epochs=5, lr=0.001, batch_size=64,
            device=device, use_dropout=False, use_batchnorm=False
        )
        print(f"Training time: {duration:.2f} sec")
        print(f"Final Test Accuracy: {acc:.4f}")
        print(f"Params: {params:,}")


def run_grid_search(
    base_widths,
    schemes,
    dataset='mnist',
    epochs=3,
    lr=0.001,
    batch_size=64,
    device='cpu',
    use_dropout=False,
    use_batchnorm=False
):
    results = {}
    for scheme_name, scheme_func in schemes.items():
        accs = []
        for base in base_widths:
            sizes = scheme_func(base)
            print(f"Testing scheme '{scheme_name}' with base width {base}: {sizes}")
            acc, duration, params = run_width_experiment(
                sizes, dataset, epochs, lr, batch_size, device, use_dropout, use_batchnorm
            )
            print(f"→ Accuracy: {acc:.4f}, Time: {duration:.2f}s, Params: {params:,}")
            accs.append(acc)
        results[scheme_name] = accs

    # Визуализация heatmap
    plot_heatmap(results, base_widths, save_path="plots/width_grid_search_heatmap.png")

    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2.1 — фиксированные ширины (3 слоя)
    # width_variants = [
    #     [64, 32, 16],      
    #     [256, 128, 64],    
    #     [1024, 512, 256],  
    #     [2048, 1024, 512]  
    # ]
    # run_basic_width_experiments(width_variants, device=device)

    # 2.2 — grid search по схемам ширины
    base_widths = [32, 64, 128]
    schemes = {
        "expanding": lambda b: [b, b*2, b*4],
        "shrinking": lambda b: [b*4, b*2, b],
        "constant": lambda b: [b, b, b],
    }
    run_grid_search(base_widths, schemes, device=device)
