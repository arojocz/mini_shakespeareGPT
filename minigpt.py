import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import requests
import time
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

batch_size = 32        
block_size = 128       
max_iters = 200   
eval_interval = 50 
learning_rate = 3e-4   
eval_iters = 50        
n_embd = 384           
n_head = 6             
n_layer = 4            
dropout = 0.2          
model_filename = 'mini_gpt_shakespeare.pth'

torch.manual_seed(1337)

print(f"Device: {device}")

# DATA LOADER & TOKENIZER

if not os.path.exists('input.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open('input.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# MODEL ARCHITECTURE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        v = self.value(x) 
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_encoding = PositionalEncoding(n_embd, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        x = self.position_encoding(tok_emb) 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, strategy='multinomial'):
        """ Standard generation: Supports Greedy and Multinomial """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if strategy == 'greedy':
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def generate_beam(self, idx, max_new_tokens, beam_width=3):
        """ Beam Search Implementation """
        # Only works with batch_size=1
        beams = idx.repeat(beam_width, 1) 
        beam_scores = torch.zeros(beam_width, device=idx.device)
        beam_scores[1:] = float('-inf') 
        
        for _ in range(max_new_tokens):
            idx_cond = beams[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            next_scores = beam_scores.unsqueeze(1) + log_probs
            next_scores_flat = next_scores.view(-1)
            
            topk_scores, topk_indices = torch.topk(next_scores_flat, beam_width)
            
            beam_indices = topk_indices.div(vocab_size, rounding_mode='floor')
            token_indices = topk_indices % vocab_size
            
            new_beams = torch.cat((beams[beam_indices], token_indices.unsqueeze(1)), dim=1)
            
            beams = new_beams
            beam_scores = topk_scores
            
        best_beam_idx = torch.argmax(beam_scores)
        return beams[best_beam_idx].unsqueeze(0)

# PLOTSS

def plot_positional_encoding():
    pe_module = PositionalEncoding(d_model=384, max_len=128)
    pe_data = pe_module.pe.squeeze(0).cpu().numpy() 
    plt.figure(figsize=(10, 6))
    plt.imshow(pe_data, aspect='auto', cmap='viridis')
    plt.title("Positional Encoding Heatmap")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar(label="Value")
    plt.savefig('positional_encoding_heatmap.png')
    plt.close()

def plot_loss_history(train_history, val_history):
    plt.figure(figsize=(12, 6))
    
    # Train loss per step
    plt.plot(train_history, label='Training Loss', alpha=0.3, color='blue', linewidth=1)
    
    # Validation loss per interval
    val_steps, val_losses = zip(*val_history)
    plt.plot(val_steps, val_losses, label='Validation Loss', color='orange', marker='o', linewidth=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('loss.png')
    plt.close()


if __name__ == '__main__':
    
    plot_positional_encoding()

    model = GPTLanguageModel()
    m = model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

    train_loss_history = [] 
    val_loss_history = []

    if os.path.exists(model_filename):
        print(f"[INFO] Loading checkpoint: {model_filename}")
        m.load_state_dict(torch.load(model_filename, map_location=device))
    
    else:
        print(f"[INFO] Starting training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        for iter in range(max_iters):
            
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                val_loss_history.append((iter, losses['val']))
            
                print("-" * 50)
                m.eval() 
                with torch.no_grad():

                    start_str = "Capulet"
                    ctx = torch.tensor(encode(start_str), dtype=torch.long, device=device).unsqueeze(0)
                    
                    # 1. Multinomial
                    out_multi = m.generate(ctx, max_new_tokens=50, strategy='multinomial')
                    print(f"[Multinomial]: {decode(out_multi[0].tolist()).replace('\n', ' ')}")
                    
                    # 2. Beam Search
                    out_beam = m.generate_beam(ctx, max_new_tokens=50, beam_width=3)
                    print(f"[Beam Search]: {decode(out_beam[0].tolist()).replace('\n', ' ')}")
                    
                    # 3. Greedy
                    out_greedy = m.generate(ctx, max_new_tokens=50, strategy='greedy')
                    print(f"[Greedy]     : {decode(out_greedy[0].tolist()).replace('\n', ' ')}")
                    
                print("-" * 50)
                m.train()
                
                torch.save(m.state_dict(), model_filename)

            # --- TRAINING STEP ---
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            
            # Record detailed training loss
            train_loss_history.append(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        plot_loss_history(train_loss_history, val_loss_history)

    # --- INTERACTIVE FINAL MODE ---
    print("\n" + "="*50)
    print("Shakespeare style chat: ")
    print("   (Multinomial vs Beam Search vs Greedy)")
    print("="*50)

    while True:
        try:
            user_input = input("\nbegin:  > ")
            if user_input.lower() in ['exit', 'quit', 'salir']: break
            if not user_input: continue

            # Preparamos el contexto (B=1)
            context_idx = torch.tensor(encode(user_input), dtype=torch.long, device=device).unsqueeze(0)
        
            m.eval()
            with torch.no_grad():
                # Cantidad de letras a generar
                num_tokens = 100 

                # 1. MULTINOMIAL
                print("\n" + "-"*15 + " [Multinomial] " + "-"*15)
                out_multi = m.generate(context_idx, max_new_tokens=num_tokens, strategy='multinomial')
                print(decode(out_multi[0].tolist()))

                # 2. BEAM SEARCH
                print("\n" + "-"*15 + " [Beam Search] " + "-"*15)
                # beam_width=5 da mejores resultados que 3, pero es un poco más lento
                out_beam = m.generate_beam(context_idx, max_new_tokens=num_tokens, beam_width=5)
                print(decode(out_beam[0].tolist()))

                # 3. GREEDY
                print("\n" + "-"*15 + " [Greedy] " + "-"*15)
                out_greedy = m.generate(context_idx, max_new_tokens=num_tokens, strategy='greedy')
                print(decode(out_greedy[0].tolist()))
            
            print("\n" + "="*60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n.ERROR: {e}")