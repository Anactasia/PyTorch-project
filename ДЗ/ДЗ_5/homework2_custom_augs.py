from datasets import CustomImageDataset
from utils import show_full_augmentation_view
from extra_augs import AddGaussianNoise, CutOut, AutoContrast
from extra_augs import RandomBlur, RandomPerspective, RandomBrightnessContrast
from torchvision import transforms

# Загрузка 5 изображений из разных классов
dataset = CustomImageDataset('data/train', transform=None)
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

# Наши кастомные аугментации
custom_augs = [
    ("Blur", RandomBlur(p=1.0)),
    ("Perspective", RandomPerspective(p=1.0)),
    ("Bright/Contrast", RandomBrightnessContrast(p=1.0))
]

# Сравнение с готовыми
extra_augs = [
    ("Noise", AddGaussianNoise(std=0.2)),
    ("CutOut", CutOut(p=1.0)),
    ("AutoContrast", AutoContrast(p=1.0))
]

# Применение и визуализация
for img, label in zip(selected_imgs, selected_labels):
    print(f"\nКласс: {dataset.get_class_names()[label]}")

    # Кастомные аугментации
    custom_aug_imgs = [aug(img) for _, aug in custom_augs]
    custom_titles = [name for name, _ in custom_augs]
    combined_custom = transforms.Compose([aug for _, aug in custom_augs])(img)
    show_full_augmentation_view(img, custom_aug_imgs, combined_custom, custom_titles)

    # Готовые аугментации (на тензоре)
    img_tensor = transforms.ToTensor()(img)
    extra_aug_imgs = [aug(img_tensor.clone()) for _, aug in extra_augs]
    extra_titles = [name for name, _ in extra_augs]
    combined_extra = transforms.Compose([aug for _, aug in extra_augs])(img_tensor)
    show_full_augmentation_view(img, extra_aug_imgs, combined_extra, extra_titles)