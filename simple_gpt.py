#Implementation of attention powered Transformer with single head attention and sub word tokenization

import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Hyperparameters
vocab_size = 64 #number of different tokens in dictionary
block_size = 8 #maximum number of tokens to be considered as one "input"
n_embed = 32 #dimension of vector after embedding

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super.__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

    def forward(self, idx):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        positional_embedding = self.token_embedding_table(torch.arange(T, device=device))
        x = token_embedding + positional_embedding
        return x



