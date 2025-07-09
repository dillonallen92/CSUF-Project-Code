import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data as data 
import math 
import copy 

# Multihead attention
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    assert d_model % num_heads == 0        # this needs to happen as d_model should be divisible by the number of heads

    self.d_model   = d_model 
    self.num_heads = num_heads

    self.d_k = d_model // num_heads        # integer division to get the dimensionality of each key, this is for the scaled dot product

    # Establish weights for each of the channels (Query, Key, Value) and Output
    self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
    self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
    self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
    self.W_o = nn.Linear(in_features=d_model, out_features=d_model)

  def scaled_dot_product_attention(self, Q, K, V, mask = None):
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # the (-2, -1) in transpose is made for switching the last 2 dimensions
    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask==0, -1e9)
    attention_probabilities = torch.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_probabilities, V)
    return output 

  def split_heads(self, x):
    batch_size, seq_length, d_model = x.size()
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

  def combine_heads(self, x):
    batch_size, _, seq_length, d_k = x.size()
    return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)

  def forward(self, Q, K, V, mask=None):
    Q = self.split_heads(self.W_q(Q))
    K = self.split_heads(self.W_k(K))
    V = self.split_heads(self.W_v(V))

    attention_output = self.scaled_dot_product_attention(Q,K,V,mask)
    output = self.W_o(self.combine_heads(attention_output))
    return output

class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff):
    super().__init__()
    self.fc1  = nn.Linear(d_model, d_ff)
    self.fc2  = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()

    pe       = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))

    # create the positional encoding functions
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe.unsqueeze(0))
  
  def forward(self, x):
    return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.feedforward    = PositionWiseFeedForward(d_model= d_model, d_ff = d_ff)
    self.norm1          = nn.LayerNorm(d_model)
    self.norm2          = nn.Layer(d_model)
    self.dropout        = nn.Dropout(dropout)

  def forward(self, x, mask):
    attention_output   = self.self_attention(x, x, x, mask)
    x                  = self.norm1(x + self.dropout(attention_output))
    feedforward_output = self.feedforward(x)
    x                  = self.norm2(x + self.dropout(feedforward_output))
    return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.self_attention  = MultiHeadAttention(d_model, num_heads)
    self.cross_attention = MultiHeadAttention(d_model, num_heads)
    self.feedforward     = PositionWiseFeedForward(d_model, d_ff)
    self.norm1           = nn.LayerNorm(d_model)
    self.norm2           = nn.LayerNorm(d_model)
    self.norm3           = nn.LayerNorm(d_model)
    self.dropout         = nn.Dropout(dropout)
  
  def forward(self, x, enc_output, src_mask, tgt_mask):
    attention_output   = self.self_attention(x, x, x, tgt_mask)
    x                  = self.norm1(x + self.dropout(attention_output))
    attention_output   = self.cross_attention(x, enc_output, enc_output, src_mask)
    x                  = self.norm2(x + self.dropout(attention_output))
    feedforward_output = self.feedforward(x)
    x                  = self.norm3(x + self.dropout(feedforward_output))

# Finally, we can make the transformer model
class Transformer(nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
    super().__init__()
    self.encoder_embedding   = nn.Embedding(src_vocab_size, d_model)
    self.decoder_embedding   = nn.Embedding(tgt_vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

    self.encoding_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    self.decoding_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    self.fully_connected_layer = nn.Linear(d_model, tgt_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def generate_mask(self, src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

  def forward(self, src, tgt):
    src_mask, tgt_mask = self.generate_mask(src, tgt)
    src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
    tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

    enc_output = src_embedded
    for enc_layer in self.encoder_layers:
      enc_output = enc_layer(enc_output, src_mask)
    
    dec_output = tgt_embedded
    for dec_layer in self.decoding_layers:
      dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
    
    output = self.fully_connected_layer(dec_output)
    return output
    

