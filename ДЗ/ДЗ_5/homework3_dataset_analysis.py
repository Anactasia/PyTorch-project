from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

# 1. Загрузка датасета
dataset = CustomImageDataset('data/train', transform=None)

# 2. Подсчет количества изображений по классам
labels = [label for _, label in dataset]
class_counts = Counter(labels)
class_names = dataset.get_class_names()

print("Количество изображений по классам:")
for cls_idx, count in class_counts.items():
    print(f"{class_names[cls_idx]}: {count}")

# 3. Сбор размеров изображений
sizes = [img.size for img, _ in dataset]  # (width, height)
widths, heights = zip(*sizes)

min_size = (min(widths), min(heights))
max_size = (max(widths), max(heights))
avg_size = (int(np.mean(widths)), int(np.mean(heights)))

print(f"Минимальный размер изображения: {min_size}")
print(f"Максимальный размер изображения: {max_size}")
print(f"Средний размер изображения: {avg_size}")

# 4. Визуализация
plt.figure(figsize=(12,5))

# Гистограмма размеров
plt.subplot(1, 2, 1)
plt.hist(widths, bins=20, alpha=0.7, label='Width')
plt.hist(heights, bins=20, alpha=0.7, label='Height')
plt.title("Распределение размеров изображений")
plt.xlabel("Размер (пиксели)")
plt.ylabel("Количество")
plt.legend()

# Гистограмма по классам
plt.subplot(1, 2, 2)
plt.bar([class_names[i] for i in class_counts.keys()], class_counts.values())
plt.title("Количество изображений по классам")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
