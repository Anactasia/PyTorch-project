from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=128, stride=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Токенизация всего текста + добавление BOS/EOS
        tokenized = tokenizer.encode(text).ids
        bos = tokenizer.token_to_id("<s>")
        eos = tokenizer.token_to_id("</s>")

        if bos is not None:
            tokenized = [bos] + tokenized
        if eos is not None:
            tokenized += [eos]

        self.examples = []
        for i in range(0, len(tokenized) - max_length, stride):
            chunk = tokenized[i:i + max_length]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

        # Добавим последний неполный кусок (если надо)
        last_start = ((len(tokenized) - max_length) // stride) * stride
        if last_start + stride < len(tokenized) - 1:
            chunk = tokenized[-max_length:]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx][:-1]
        y = self.examples[idx][1:]
        return x, y
