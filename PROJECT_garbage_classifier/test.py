import torch
from utils.dataset import get_dataloaders
from models.cnn import SimpleCNN
from utils.metrics import compute_metrics, plot_confusion_matrix

def test_model(data_dir, model_path='PROJECT_garbage_classifier/models/best_model_ResNet18.pth', batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    _, _, test_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Метрики и сохранение
    save_metrics_to = 'PROJECT_garbage_classifier/metrics_output'
    report, cm = compute_metrics(all_labels, all_preds, class_names, save_path=save_metrics_to)
    plot_confusion_matrix(cm, class_names, save_path=save_metrics_to)

if __name__ == '__main__':
    test_model(data_dir='PROJECT_garbage_classifier/data')
