from datasets import CustomImageDataset
from utils import show_single_augmentation, show_multiple_augmentations, show_full_augmentation_view
from torchvision import transforms

# 1. Загрузка датасета
dataset = CustomImageDataset('data/train', transform=None)

# 2. Выбираем 5 изображений из разных классов
selected_imgs = []
selected_labels = []
classes_seen = set()

for img, label in dataset:
    if label not in classes_seen:
        selected_imgs.append(img)
        selected_labels.append(label)
        classes_seen.add(label)
    if len(classes_seen) == 5:
        break

# 3. Создаем стандартные аугментации
standard_augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
]

from utils import show_full_augmentation_view  # ты это уже импортировал выше

# 4. Визуализация всех аугментаций для каждого изображения
for img, label in zip(selected_imgs, selected_labels):
    class_name = dataset.get_class_names()[label]
    print(f"\nКласс: {class_name}")

    aug_imgs = []
    aug_titles = []

    # Применяем каждую аугментацию отдельно
    for name, aug in standard_augs:
        pipeline = transforms.Compose([aug, transforms.ToTensor()])
        aug_img = pipeline(img)
        aug_imgs.append(aug_img)
        aug_titles.append(name)

    # Применяем все аугментации вместе
    combined_pipeline = transforms.Compose([*(aug for _, aug in standard_augs), transforms.ToTensor()])
    combined_img = combined_pipeline(img)

    # Показываем 3 строки: оригинал | все отдельно | все вместе
    show_full_augmentation_view(img, aug_imgs, combined_img, aug_titles)


# 5. Визуализируем все аугментации сразу на одном изображении
augmented_imgs = []
titles = []
for name, aug in standard_augs:
    aug_pipeline = transforms.Compose([aug, transforms.ToTensor()])
    for img in selected_imgs:
        aug_img = aug_pipeline(img)
        augmented_imgs.append(aug_img)
        titles.append(name)
    break  # чтобы показать пример только для первого изображения

show_multiple_augmentations(selected_imgs[0], augmented_imgs[:len(standard_augs)], titles[:len(standard_augs)])

# 6. Пайплайн со всеми аугментациями вместе
combined_aug = transforms.Compose([*(aug for _, aug in standard_augs), transforms.ToTensor()])
for img in selected_imgs:
    combined_img = combined_aug(img)
    show_single_augmentation(img, combined_img, title="Все аугментации вместе")
