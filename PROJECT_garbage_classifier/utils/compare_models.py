import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from models.cnn import SimpleCNN
from models.pretrained_model import get_pretrained_model
from utils.dataset import get_dataloaders
from tabulate import tabulate
import numpy as np

# Настройки 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'PROJECT_garbage_classifier/data'
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
num_classes = len(class_names)
batch_size = 32

# Загрузка датасета
_, _, test_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)

# Модели
models_info = {
    'SimpleCNN': {
        'model': SimpleCNN(num_classes=num_classes),
        'path': 'PROJECT_garbage_classifier/models/best_model_SimpleCNN.pth'
    },
    'ResNet18': {
        'model': get_pretrained_model('resnet18', num_classes=num_classes),
        'path': 'PROJECT_garbage_classifier/models/best_model_ResNet18.pth'
    }
}

# Функция оценки
def evaluate_model(model, name):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n{name} Accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n", tabulate([[name] + [f"{report[c]['f1-score']:.2f}" for c in class_names]] + 
                         [[""] + [""] * len(class_names)], 
                         headers=["Model"] + class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Печать confusion matrix в текстовом формате
    cm_table = tabulate(cm, headers=class_names, showindex=class_names, tablefmt="grid")
    print(f"\n Confusion Matrix — {name} (табличный вид):")
    print(cm_table)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.show()

    return acc, report


if __name__ == "__main__":
    results = []

    for model_name, info in models_info.items():
        model = info['model']
        model.load_state_dict(torch.load(info['path'], map_location=device))
        acc, report = evaluate_model(model, model_name)
        results.append({
            'name': model_name,
            'accuracy': acc,
            'f1_macro': report['macro avg']['f1-score'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall']
        })

    #Таблица сравнения
    print("\nСводная таблица по метрикам:")
    print(tabulate(
        [[r['name'], f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}"]
         for r in results],
        headers=["Model", "Accuracy", "F1 (macro)", "Precision", "Recall"]
    ))
