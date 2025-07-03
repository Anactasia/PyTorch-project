import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f" Сохранено: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_heatmap(results, base_widths, save_path=None, title='Grid Search Accuracy Heatmap'):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.array(list(results.values())),
        annot=True,
        xticklabels=base_widths,
        yticklabels=list(results.keys()),
        cmap="viridis",
        cbar_kws={'label': 'Test Accuracy'}
    )
    plt.xlabel('Base Width')
    plt.ylabel('Scheme')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_weight_distribution(model, save_path=None):
    weights = []
    for param in model.parameters():
        if param.requires_grad and len(param.shape) > 1:  # только веса, не biases
            weights.append(param.data.cpu().numpy().flatten())

    import numpy as np
    all_weights = np.concatenate(weights)

    plt.figure(figsize=(8, 4))
    plt.hist(all_weights, bins=100, alpha=0.75, color='steelblue')
    plt.title("Weight Distribution")
    plt.xlabel("Weight values")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved weight distribution to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_weight_distributions_comparison(models, labels, save_dir="plots", experiment_name="weights_comparison"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Получаем имена слоев (будем считать только слои с весами)
    layer_names = []
    for name, param in models[0].named_parameters():
        if 'weight' in name:
            layer_names.append(name)

    # Для каждого слоя рисуем гистограмму весов всех моделей на одном графике
    for layer_name in layer_names:
        plt.figure(figsize=(8, 6))
        for model, label in zip(models, labels):
            # Извлекаем веса слоя
            weights = None
            for name, param in model.named_parameters():
                if name == layer_name:
                    weights = param.detach().cpu().numpy().flatten()
                    break
            if weights is None:
                continue
            
            # Рисуем гистограмму (прозрачность alpha для удобства сравнения)
            plt.hist(weights, bins=50, alpha=0.5, density=True, label=label)
        
        plt.title(f"Weight distribution comparison for layer: {layer_name}")
        plt.xlabel("Weight value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        
        # Сохраняем график
        filename = f"{experiment_name}_{layer_name.replace('.', '_')}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()