from torchvision import transforms

class AugmentationPipeline:
    def __init__(self):
        self.augs = {}

    def add_augmentation(self, name, aug):
        self.augs[name] = aug

    def remove_augmentation(self, name):
        if name in self.augs:
            del self.augs[name]

    def apply(self, image):
        composed = transforms.Compose(list(self.augs.values()))
        return composed(image)

    def get_augmentations(self):
        return list(self.augs.keys())


from datasets import CustomImageDataset
from utils import show_single_augmentation
from torchvision import transforms

# Загрузка датасета
dataset = CustomImageDataset('data/train', transform=None)

# Создаем пайплайны
light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=1.0))

medium_pipeline = AugmentationPipeline()
medium_pipeline.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=1.0))
medium_pipeline.add_augmentation('RandomRotation', transforms.RandomRotation(degrees=15))

heavy_pipeline = AugmentationPipeline()
heavy_pipeline.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=1.0))
heavy_pipeline.add_augmentation('RandomRotation', transforms.RandomRotation(degrees=30))
heavy_pipeline.add_augmentation('ColorJitter', transforms.ColorJitter(brightness=0.5, contrast=0.5))

# Пример применения
for i, (img, label) in enumerate(dataset):
    if i >= 5:
        break
    print(f"Класс: {dataset.get_class_names()[label]}")

    light_img = light_pipeline.apply(img)
    show_single_augmentation(img, light_img, title="Light Pipeline")

    medium_img = medium_pipeline.apply(img)
    show_single_augmentation(img, medium_img, title="Medium Pipeline")

    heavy_img = heavy_pipeline.apply(img)
    show_single_augmentation(img, heavy_img, title="Heavy Pipeline")
