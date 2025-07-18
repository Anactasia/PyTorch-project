# Классификация мусора по типу материала с помощью сверточной нейросети

Автоматическая классификация изображений мусора с помощью сверточных нейронных сетей (CNN) и предобученных моделей (ResNet). Проект реализует полный ML-пайплайн — от предобработки данных до визуализации результатов и запуска модели через графический интерфейс.

---

## Датасет

Используется датасет с изображениями мусора, взятый с Kaggle:  
[https://drive.google.com/drive/folders/1w8rbdJaHYE5g-GkpPZ1K03hzyp9arSjs?usp=sharing](гугл диск)

папку "data" необходимо добавить в папку PROJECT_garbage_classifier


## Структура проекта

```
PROJECT_garbage_classifier/
├── metrics_output/
│ ├── classification_report.txt # Сохранённый отчет по метрикам
│ ├── confusion_matrix_ResNet18.png # Матрица ошибок для ResNet
│ └── confusion_matrix_SimpleCNN.png # Матрица ошибок для собственной CNN
│
├── models/
│ ├── best_model_ResNet18.pth # Сохранённые веса предобученной модели
│ ├── best_model_SimpleCNN.pth # Сохранённые веса собственной CNN
│ ├── cnn.py # Реализация собственной сверточной нейросети
│ └── pretrained_model.py # Подключение и адаптация ResNet18
│
├── utils/
│ ├── compare_models.py # Сравнение качества разных моделей
│ ├── data_separation.py # Разделение изображений на train/val/test
│ ├── plot_utils.py # Функции для построения графиков
│ ├── dataset.py # DataLoader'ы, трансформации, загрузка датасета
│ └── metrics.py # Расчет метрик и визуализация результатов
│
├── app.py # GUI-интерфейс на tkinter для предсказаний
├── train.py # Обучение моделей
├── test.py # Оценка модели на тестовой выборке
├── requirements.txt # Список зависимостей
└── README.md # Документация проекта
```