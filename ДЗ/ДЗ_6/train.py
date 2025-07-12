import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from model import GeneratorTransformer  # твой файл с моделью
from dataset import TextDataset          # твой файл с датасетом

# Параметры обучения
BATCH_SIZE = 1
MAX_LENGTH = 128
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def train():
    tokenizer = Tokenizer.from_file("transformer_basics/ДЗ_6/mistral_tokenizer.json")
    text = load_text("data/2.txt")

    dataset = TextDataset(text, tokenizer, max_length=MAX_LENGTH, stride=MAX_LENGTH//2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GeneratorTransformer(vocab_size=tokenizer.get_vocab_size(), max_length=MAX_LENGTH, tokenizer=tokenizer)
    model.to(DEVICE)

    # ✅ Загрузка предыдущей обученной модели
    start_epoch = 11
    model.load_state_dict(torch.load(f"checkpoint_epoch{start_epoch - 1}.pt"))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + 10):  # 11 → 15
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")

if __name__ == "__main__":
    train()
