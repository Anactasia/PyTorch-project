from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_metrics(y_true, y_pred, class_names=None, save_path=None, print_report=True):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, output_dict=False)
    matrix = confusion_matrix(y_true, y_pred)

    if print_report:
        print("\n📊 Classification Report:\n")
        print(report)
        print("\n🧩 Confusion Matrix:\n")
        print(matrix)

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Сохраняем текстовый отчёт
        with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(matrix))

    return report, matrix

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))

    plt.show()
