from training import train_model
from utils.model_utils import save_model
from utils.model_utils import count_parameters
import json
import os

def run_experiment(model, train_loader, test_loader, epochs, lr, device, weight_decay=0.0):
    history = train_model(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        weight_decay=weight_decay
    )
    return history



import json
import os

def save_experiment_results(
    save_dir,
    experiment_name,
    history,
    duration,
    model,
    depth=None,
    widths=None,
    hidden_size=128,
    use_dropout=False,
    use_batchnorm=False,
    epochs=10,
    lr=0.001,
    batch_size=64,
    device='cpu'
):
    os.makedirs(save_dir, exist_ok=True)

    # Основная информация об эксперименте
    experiment_info = {
        "depth": depth,
        "widths": widths,
        "hidden_size": hidden_size,
        "use_dropout": use_dropout,
        "use_batchnorm": use_batchnorm,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "device": device,
    }

    # Метрики
    metrics = {
        "train_accuracy_per_epoch": history['train_accs'],
        "test_accuracy_per_epoch": history['test_accs'],
        "train_loss_per_epoch": history['train_losses'],
        "test_loss_per_epoch": history['test_losses'],
        "final_train_accuracy": history['train_accs'][-1],
        "final_test_accuracy": history['test_accs'][-1],
        "training_time_sec": duration,
        "num_parameters": count_parameters(model),
    }

    results = {
        "experiment_info": experiment_info,
        "metrics": metrics
    }

    save_path = os.path.join(save_dir, f"{experiment_name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {save_path}")
