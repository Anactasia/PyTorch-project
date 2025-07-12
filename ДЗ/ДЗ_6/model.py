import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]  # по (B, n_heads, T, d_head)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, n_heads, T, T)
        
        # Маска — триангл (чтобы нельзя было видеть будущие токены)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        # Self-attention + residual
        x = x + self.self_attn(self.ln1(x), mask)
        # Feed-forward + residual
        x = x + self.ff(self.ln2(x))
        return x


class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, d_ff=1024, max_length=128, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding + positional embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_length, d_model))
        
        # Декодерные слои
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # токены eos, bos
        self.eos_token_id = tokenizer.token_to_id('</s>') if tokenizer else vocab_size - 1
        self.bos_token_id = tokenizer.token_to_id('<s>') if tokenizer else 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.max_length, "Sequence length must be <= max_length"
        
        # Токен + позиционное эмбеддинги
        token_embeddings = self.token_emb(input_ids)  # (B, T, d_model)
        pos_embeddings = self.pos_emb[:, :T, :]       # (1, T, d_model)
        x = token_embeddings + pos_embeddings
        
        # Создаем маску для автогрессионной генерации (триангл)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200):
        self.eval()
        with torch.no_grad():
            input_ids = torch.tensor([self.tokenizer.encode(prompt).ids]).to(self.device)
            generated = input_ids.clone()
            for _ in range(max_out_tokens):
                input_ids_trim = generated[:, -context_len:]
                outputs = self(input_ids_trim)
                next_token_logits = outputs[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.eos_token_id:
                    break
            return self.tokenizer.decode(generated[0].tolist())
    
    def generate_beam(self, prompt, context_len=50, beam_size=3, max_out_tokens=100):
        self.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], device=self.device)

            beams = [(input_ids, 0.0)]  # список: (текущая последовательность, log-проб)

            for _ in range(max_out_tokens):
                new_beams = []
                for seq, score in beams:
                    input_trimmed = seq[:, -context_len:]
                    logits = self(input_trimmed)  # (1, T, vocab_size)
                    next_token_logits = logits[0, -1, :]
                    probs = torch.log_softmax(next_token_logits, dim=-1)

                    topk = torch.topk(probs, beam_size)

                    for token_id, token_logprob in zip(topk.indices, topk.values):
                        new_seq = torch.cat([seq, token_id.view(1, 1)], dim=1)
                        new_score = score + token_logprob.item()
                        new_beams.append((new_seq, new_score))

                # Оставляем top beam_size вариантов
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

                # Если в каком-то из beam встретился <eos>, прерываем
                if any(seq[0, -1].item() == self.eos_token_id for seq, _ in beams):
                    break

            # Возвращаем самый вероятный
            best_sequence = beams[0][0][0].tolist()
            return self.tokenizer.decode(best_sequence)

