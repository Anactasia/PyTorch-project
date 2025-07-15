import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from models.pretrained_model import get_pretrained_model

# Настройки
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'PROJECT_garbage_classifier/models/best_model_ResNet18.pth'
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Загрузка модели
model = get_pretrained_model('resnet18', num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Преобразования
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Предсказание
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Выбор изображения
def select_image():
    temp = Toplevel()
    temp.withdraw()
    file_path = filedialog.askopenfilename(
        parent=temp,
        title="Выберите изображение мусора",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    temp.destroy()

    if file_path:
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk

        label = predict(file_path)
        result_label.config(text=f"Класс: {label}", fg="#00ff99")

# Темный стиль
bg_color = "#1e1e1e"
fg_color = "#ffffff"
accent_color = "#00ff99"
btn_color = "#2e2e2e"
font_main = ("Segoe UI", 12)
font_title = ("Segoe UI", 18, "bold")

# Окно
root = tk.Tk()
root.title("🗑️ Сортировка мусора")
root.configure(bg=bg_color)

# Центрирование
window_width = 450
window_height = 580
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(False, False)

# Заголовок
title_label = tk.Label(root, text="Классификатор мусора", font=font_title, bg=bg_color, fg=accent_color)
title_label.pack(pady=20)

# Кнопка
btn = tk.Button(root, text="📂 Загрузить изображение", command=select_image, font=font_main,
                bg=btn_color, fg=fg_color, activebackground=accent_color, activeforeground="#000000", bd=0, padx=10, pady=5)
btn.pack(pady=10)

# Изображение
panel = tk.Label(root, bg=bg_color)
panel.pack(pady=20)

# Результат
result_label = tk.Label(root, text="Ожидание загрузки...", font=("Segoe UI", 14), bg=bg_color, fg="#cccccc")
result_label.pack(pady=10)

# Запуск
root.mainloop()
