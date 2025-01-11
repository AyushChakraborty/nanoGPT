from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken as tkn

#-------------------------
#here the namings are done as per the openai/HF implementation so that the weights can be ported to this 
#implementation of GPT2 easily


if torch.mps.is_available():
    device = torch.device("mps")  #change to cuda if available
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@dataclass  #this decorator automatically generates an __init__ method for the class with the
#attributes as variables defined under the class for which the decorator is called
class GPTConfig:
    block_size: int = 1024     #number of tokens in the context window
    vocab_size: int = 50257    #number of unique tokens in the vocabulary, 50000 BPE megres, 256 byte tokens, 1 <|endoftext|> token 
    n_layer: int = 12          #number of blocks in the model
    n_head: int = 12           #number of heads in the multiheadattention
    n_embd: int = 768



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0   #making sure that the head dimension is a factor of the embedding dimension
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  #linear layer for the attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)   
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        #registering the bias as a buffer, which is a tensor that is not updated during the training process
        #its actually a mask rather than a bias but again named this way following the openai/HF implementation
        #here it is reshaped to a 4d tensor, where 0th dim is the batch dim, 1st dim is the head dim, so
        #for each batch, there are many heads of attention grids, and the 2nd and 3rd dim make up the attention
        #grid itself


    def forward(self, x):
        B, T, C = x.shape      #batch size, tokens in the context window, embedding vector for each token
        qkv = self.c_attn(x)   #(B, T, 3*config.n_embd) where C itself is config.n_embd so its 3*C
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        #here, instead of adding the vectors obtained from diff heads, we concatenate them and hence
        #we split the vectors n_head parts, each for one head of the attention block, and each having
        #C//n_head which is head_size as number of elements
        q = q.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  #essentially doing (k dot q )/ sqrt(d_k), (B, n_head, T, T)
        #so the (T, T) attention grid for each head for each batch is obtained

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  #masking phase, also :T is present as during generation, the context window
        #can vary from 1 to T tokens

        y = att @ v  # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        #here we get the weighted value vector for wach token for each batch and for each head

        y = y.transpose(1, 2).contiguous().view(B, T, C)  #here we concatenate those value vectors for a
        #token obtained from all the diff heads across all the tokens and across all the batches, so now 
        #C has size n_head*head_size 
        y = self.c_proj(y)                                #passing through the linear layer

        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  #GELU or gaussian error linear unit is a softer version of RELU, where one of its advantage is that there is always some gradient contribution hence the issue of dead gradients is solved
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  #making the embedding table of vocab_size, n_embd
            wpe = nn.Embedding(config.block_size, config.n_embd),  #making the positional embedding table of block_size, n_embd
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  #blocks
            ln_f = nn.LayerNorm(config.n_embd)                                  #final layer norm layer
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  #the final linear layer, to get the logits


    def forward(self, idx):  #used for generation of new tokens
        #idx has shape (B, T) where T <= config.block_size and its obviously varible in this case
        B, T = idx.shape
        assert T <= self.config.block_size, f"cannot forward sequence of length {T} because GPT2 is limited to block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  #(T)
        pos_emb = self.transformer.wpe(pos)  #(T, n_embd)
        tok_emb = self.transformer.wte(idx)      #(B, T, n_embd)
        x = tok_emb + pos_emb  #(B, T, n_embd), the same tokens in embedding space but now infused with 
        #positional information

        #since the goal of this method is to generate, we have to step through the blocks, and thats what we do now
        for b in self.transformer.h:
            x = b(x)
        
        logits = self.transformer.ln_f(x)  #(B, T, vocab_size), passing it through the final linear layer so as to obtain the logits from the last token
        logits = self.lm_head(logits)      #(B, T, vocab_size), the logits for the next token
        return logits    
        #also since we feed the generation step also in batches it means that multiple sequences can
        #be generated at once


    @classmethod
    def from_pretrained(cls, model_type):
        """loads the pretrained GPT-2 model from huggingface, so that we can skip the training phase, and 
        load the params for the model, and this works as this is the same model arch as the one huggingface
        implemented"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} 
        print(f'loading {model_type} model...')

        config_args = {
            'gpt2' : dict(n_layer = 12, n_head = 12, n_embd = 768),          #124M params
            'gpt2-medium' : dict(n_layer = 24, n_head = 16, n_embd = 1024),  #345M params
            'gpt2-large' : dict(n_layer = 36, n_head = 20, n_embd = 1280),   #774M params
            'gpt2-xl' : dict(n_layer = 48, n_head = 25, n_embd = 1600)       #1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)   #own implemented model

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        #now the .attn.bias is a result of the heirarchical naming of torch nn.modules, since
        #self.atnn = CausalSelfAttention(config) is called within Block, and Block is called within GPT, 
        #so keys are named as transformer.h.0.attn.bias and that number can change based on which block
        #we refer to, but to just get the bias/mask tensor, we can use .attn.bias

        #so this naming is just an artifact of the torch nn.Module heirarchy, and not a part of the model

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.masked_bias')]  #in the HF implementation, the bias is named as masked_bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
 

#-------------------------
num_return_sequences = 3
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()         #putting it in eval, altogether not necessary as we are not using dropout or batchnorm
model.to(device)     #this sets up the model

#tokenising the input 
enc = tkn.get_encoding('gpt2')                                #loads the encoding of gpt2 
tokens = enc.encode("Hello, I'm a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)               #(12, ), as seen from the tiktokenizer app, chk it out :)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  #(3, 12), so that we can generate 3 sequences at once
idx = tokens.to(device)                                       #putting it on the device

#generating !
#here idx is (num_return_sequences, T), where T is the number of tokens in the input sequence, and num_return_sequences is number of batches
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
elif torch.mps.is_available():
    torch.mps.manual_seed(42)

while idx.size(1) < max_length:
    with torch.no_grad():
        idx = idx[:, -max_length:]          #trimming the input sequence to the max_length
        logits = model(idx)                 #(num_return_sequences, T, vocab_size), where again, T is the number of tokens in the input sequence, and varies
        logits = logits[:, -1, :]           #(num_return_sequences, vocab_size),  get only the last tokens from each of the batches, as what we essentially have now at this point is a bigram problem at hand
        probs = F.softmax(logits, dim=-1)   #(num_return_sequences, vocab_size), the probabilities of the next token

        #doing top-k sampling which is huggingface default
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  #(num_return_sequences, 50), the top 50 tokens and their probabilities
        ix = torch.multinomial(topk_probs, num_samples=1)           #(num_return_sequences, 1), the indices of the tokens sampled from the top 50 tokens from each one of the batches
        idxcol = torch.gather(topk_indices, -1, ix)                   #(num_return_sequences, 1), the actual token indices sampled from the top 50 tokens
        idx = torch.cat((idx, idxcol), dim=1)                          #(num_return_sequences, T+1), the new token indices for each of the batches
        

# print the generated text
for s in range(num_return_sequences):
    tokens = idx[s, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(":: ", decoded)