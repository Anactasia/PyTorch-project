import time
import tracemalloc
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Размеры для эксперимента
sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
results = []

# Стандартные легкие аугментации
basic_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

for size in sizes:
    print(f"\nТестируем размер: {size}")

    # Засекаем время и память
    tracemalloc.start()
    start_time = time.time()

    dataset = CustomImageDataset('data/train', transform=basic_transforms, target_size=size)
    loader = DataLoader(dataset, batch_size=10)

    # Обрабатываем 100 изображений
    count = 0
    for imgs, _ in loader:
        count += imgs.size(0)
        if count >= 100:
            break

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_time = end_time - start_time
    memory_mb = peak / 1024 / 1024

    print(f"Время: {elapsed_time:.2f} сек | Память: {memory_mb:.2f} MB")
    results.append((size[0], elapsed_time, memory_mb))




# Построим графики
sizes_px = [r[0] for r in results]
times = [r[1] for r in results]
memory = [r[2] for r in results]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(sizes_px, times, marker='o')
plt.title('Время vs Размер')
plt.xlabel('Размер изображения (px)')
plt.ylabel('Время (сек)')

plt.subplot(1, 2, 2)
plt.plot(sizes_px, memory, marker='s', color='orange')
plt.title('Память vs Размер')
plt.xlabel('Размер изображения (px)')
plt.ylabel('Память (MB)')

plt.tight_layout()
plt.show()
