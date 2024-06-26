import torch
import torch.nn as nn
from torch.nn import functional as F
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''Hyperparameters'''
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 #maximum number of tokens to be considered as "input" for predictions
n_embed = 384 #dimension of vector after embedding
n_blocks = 6 #number of sequential attention blocks
num_heads = 6

max_iters = 1000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
dropout = 0.2
#Preparing Environment for training
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # Maps inputs x and targets y in an autoregerssive manner for each block in the batch
        # Token position 0 has an output of token position 1; # Tokens position 0 to 1 have an output of token position 2;
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

'''Neural Network Architecture'''

class EmbeddingBlock(nn.Module):
    #Computing the token and positional embedding for each token in a block
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

    def forward(self, idx):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx) #Output is (B, T, C)
        positional_embedding = self.token_embedding_table(torch.arange(T, device=device)) #Output is (T, C)
        x = token_embedding + positional_embedding #Output is (B, T, C)
        return x

class SingleHeadAttentionBlock(nn.Module):
    #Communication followed by computation i.e. Attention block followed by FeedForward MLP
    #Head_size is the dimension of the otuput of the key, query, value layers
    #We do not use bias
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Input size (Batch size, Time step, Channels)
        #Output size (Batch size, Time step, Head size)

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        #Calculation of affinities between each token using the values from the key and query, "implementation of self attention" because current tokens
        #Only get information from previous tokens (triangular inferior matrix)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        #The input passes by another layer "value" and we multiply the tokens by the weighted averages
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiheadAttentionBlock(nn.Module):
    #We will apply multiple heads of attention in parrallel, the output of these heads will be concatenated
    #We then apply projector layer followed by a dropout layer

    def __init__(self,  num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttentionBlock(head_size) for _ in range(num_heads)])
        self.project = nn.Linear(head_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.project(out)
        out = self.dropout(out)
        return(out)

class FeedForwardBlock(nn.Module):
    #Multiple Layer Perceptron
    def __init__(self, n_embed):
        super().__init__()
        # linear (input : n_embed, output: n_embed), relu, linear(input : n_embed, output: n_embed), dropout
        self.MLP = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.MLP(x)
        
class TransformerBlock(nn.Module):
    #Commmunication followed by computation i.e. Attention followed by Feed Forward block
    #We add layer normalization and residual connections to improve results
    def __init__(self, n_embed):
        super().__init__()
        self.MultiHeadAttentionBlock = MultiheadAttentionBlock(n_embed, num_heads)
        self.FeedForwardBlock = FeedForwardBlock(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.ln1(self.MultiHeadAttentionBlock(x))
        x = x + self.ln2(self.FeedForwardBlock(x))
        return(x)

class OutputBlock(nn.Module):
    #Final output block contains layer normalization layer and linear (input: n_embed, output: vocab_size)
    def __init__(self, n_embed):
        super().__init__()
        self.normalization_layer = nn.LayerNorm(n_embed)
        self.linear_layer = nn.Linear(n_embed, vocab_size)
        
    def forward(self, x):
        x = self.normalization_layer(x)
        x = self.linear_layer(x)
        return(x)

class LanguageModel(nn.Module):
    #We create the final model which contains the connection of the embedding transformer and output blocks
    def __init__(self):
        super().__init__()
        self.EmbeddingBlock = EmbeddingBlock()
        #We associate multiple Transformer blocks in series, using list comprehension and unpacking the list with the asterisk
        self.TransformerBlocks = nn.Sequential(*[TransformerBlock(n_embed) for _ in range(n_blocks)])
        self.OutputBlock = OutputBlock(n_embed)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        x = self.EmbeddingBlock(idx) #(B, T, C)
        B, T, C = x.shape
        x = self.TransformerBlocks(x) #(B, T, C)
        B, T, C = x.shape
        logits = self.OutputBlock(x) #(B, T, Vocab size)
        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = LanguageModel()
m = model.to(device)
print("Training is using " + device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_time = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) #Set the gradient to zero
    loss.backward() #Apply back propagation
    optimizer.step() #Update the parameters of the model

end_time = time.time()

time_elapsed = end_time - start_time 

print(str(time_elapsed/60) + "m:" + str(time_elapsed//60) + "s elapsed")

# generate from the model
#For sentencepiece context = sp.encode("\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))