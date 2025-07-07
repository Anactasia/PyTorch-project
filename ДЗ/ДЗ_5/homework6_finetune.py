import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import CustomImageDataset
import matplotlib.pyplot as plt

# ⚙️ Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧾 Аугментации и трансформы
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 📁 Датасеты и загрузчики
train_dataset = CustomImageDataset('data/train', transform=transform)
val_dataset = CustomImageDataset('data/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_classes = len(train_dataset.get_class_names())

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Тренировка
for epoch in range(10):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # Валидация
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / total)

    print(f"Epoch {epoch+1}: Train Acc={train_accuracies[-1]:.2f}, Val Acc={val_accuracies[-1]:.2f}")

#  Графики
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
