import torch
from model import GeneratorTransformer
from tokenizers import Tokenizer

def chat():
    tokenizer = Tokenizer.from_file("transformer_basics/ДЗ_6/mistral_tokenizer.json")
    model = GeneratorTransformer(vocab_size=tokenizer.get_vocab_size(), tokenizer=tokenizer)
    model.load_state_dict(torch.load("checkpoint_epoch19.pt"))  # последний чекпоинт
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    while True:
        prompt = input("Вы: ")
        if prompt.lower() == 'quit':
            break
        response = model.generate_beam(prompt, context_len=50, beam_size=3, max_out_tokens=100)

        print(f"Бот: {response}")

if __name__ == "__main__":
    chat()
