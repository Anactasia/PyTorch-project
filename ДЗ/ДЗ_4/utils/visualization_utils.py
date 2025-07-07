import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_history(history):
    """Визуализирует историю обучения"""
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
    plt.show()




def compare_models(fc_history, cnn_history):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show() 


def plot_confusion_matrix(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.show()



def plot_gradient_norms(grad_norms):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(grad_norms, marker='o')
    plt.title("Gradient Norms over Training")
    plt.xlabel("Batch number")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.show()



def visualize_feature_maps(model, image, device, max_layers=3):
    import matplotlib.pyplot as plt
    import torch.nn as nn

    model.eval()
    image = image.unsqueeze(0).to(device)

    outputs = []
    x = image

    # Только Conv2d слои
    conv_layers = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for sublayer in layer:
                conv_layers.append(sublayer)
        else:
            conv_layers.append(layer)

    for layer in conv_layers:
        if isinstance(layer, nn.Conv2d):
            x = layer(x)
            outputs.append(x.cpu())
        else:
            try:
                x = layer(x) 
            except:
                break  

        if len(outputs) >= max_layers:
            break

    # Визуализация карт признаков
    for i, fmap in enumerate(outputs):
        fig, axes = plt.subplots(1, min(8, fmap.shape[1]), figsize=(15, 4))
        for j in range(min(8, fmap.shape[1])):
            axes[j].imshow(fmap[0, j].detach().numpy(), cmap='viridis')
            axes[j].axis('off')
        plt.suptitle(f'Feature Maps after Conv Layer {i+1}')
        plt.show()

