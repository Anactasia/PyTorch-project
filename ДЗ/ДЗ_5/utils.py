import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def to_tensor_if_needed(img):
    if isinstance(img, Image.Image):
        return transforms.ToTensor()(img)
    return img

def show_full_augmentation_view(original_img, aug_imgs, combined_img, aug_titles):
    """
    Показывает оригинал, 5 отдельных аугментаций и одну комбинированную в 3 строках.
    """
    # Преобразуем в тензоры
    original_img = to_tensor_if_needed(original_img)
    aug_imgs = [to_tensor_if_needed(img) for img in aug_imgs]
    combined_img = to_tensor_if_needed(combined_img)

    # Подготовка
    resize = transforms.Resize((128, 128))
    original_img = resize(original_img)
    aug_imgs = [resize(img) for img in aug_imgs]
    combined_img = resize(combined_img)

    fig, axes = plt.subplots(3, max(len(aug_imgs), 1), figsize=(len(aug_imgs) * 2, 6))

    # Первая строка — оригинал по центру
    for ax in axes[0]:
        ax.axis('off')
    axes[0][len(aug_imgs) // 2].imshow(original_img.permute(1, 2, 0))
    axes[0][len(aug_imgs) // 2].set_title("Оригинал")

    # Вторая строка — отдельные аугментации
    for i, img in enumerate(aug_imgs):
        axes[1][i].imshow(img.permute(1, 2, 0))
        axes[1][i].axis('off')
        axes[1][i].set_title(aug_titles[i])

    # Третья строка — комбинированная
    for ax in axes[2]:
        ax.axis('off')
    axes[2][len(aug_imgs) // 2].imshow(combined_img.permute(1, 2, 0))
    axes[2][len(aug_imgs) // 2].set_title("Все вместе")

    plt.tight_layout()
    plt.show()

def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    
    # Преобразуем в Tensor, если это PIL
    if isinstance(original_img, Image.Image):
        original_img = transforms.ToTensor()(original_img)
    if isinstance(augmented_img, Image.Image):
        augmented_img = transforms.ToTensor()(augmented_img)

    # Resize тензоры
    original_img = resize_transform(original_img)
    augmented_img = resize_transform(augmented_img)

    orig_np = original_img.numpy().transpose(1, 2, 0)
    aug_np = augmented_img.numpy().transpose(1, 2, 0)

    # Отображение
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(np.clip(orig_np, 0, 1))
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    ax2.imshow(np.clip(aug_np, 0, 1))
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    resize_transform = transforms.Resize((128, 128), antialias=True)

    # Обеспечиваем совместимость: превращаем PIL в Tensor
    if isinstance(original_img, Image.Image):
        original_img = transforms.ToTensor()(original_img)
    original_img = resize_transform(original_img)

    orig_np = original_img.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        if isinstance(aug_img, Image.Image):
            aug_img = transforms.ToTensor()(aug_img)
        aug_img = resize_transform(aug_img)

        aug_np = aug_img.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()