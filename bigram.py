import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
device = torch.device("mps" if torch.mps.is_available() else "cpu")  #from now on, will utilize the GPU
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
n_block_layers = 4
# ----------------

torch.manual_seed(1337)

#load the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#preprocess the data
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]  #take in a string, give out its integer representation
decode = lambda list: ''.join([itos[i] for i in list] ) #take in a list of integers, give out its chars

#convert the text to tensor
data = torch.tensor(encode(text), dtype=torch.long, device=device) 
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#function to get a batch of data
def get_batch(name):
    data = train_data if name == 'train' else val_data
    ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
    context = torch.stack([data[i: i+block_size] for i in ix])
    target = torch.stack([data[i+1: i+1+block_size] for i in ix])
    context, target = context.to(device), target.to(device)
    return context, target


#defininig a single head of the attention block
class Head(nn.Module):
    """one head of attention block"""

    def __init__ (self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, emb_dim = x.shape
        k  = self.key(x)
        q = self.query(x)
        v = self.value(x)

        #computing the attention grid per batch
        wei = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)   #######
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  #the :T is needed
        #to handle variable context size, as the context size is not fixed
        wei = F.softmax(wei, dim=2)  #attention grid per batch (B, T, T)

        v = self.value(x)

        out = wei @ v   # (B, T, head_size)
        return out   # the tensor of matrices, where each matrix has the delta 
        #context that is to be added to each of the token vectors, so as to get the context
        #rich representation of the token


#defining a multi-head attention block, and its also helpful to have multiple heads
#as the tokens would have a lot to talk to each other about, like where are the vowels, 
#which are adjectives of a noun, and many more things, some of them we might not even
#anticipate, but through training the model would learn to attend to the right tokens
class MultiHeadAttention(nn.Module):
    '''multiple heads of self attention in parallel'''
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        #these have deltaE but in the reduced dimension
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=2)  #concat all the lower dim deltaE
        #to get the final deltaE of embedding space dim for a certain token
        return self.proj(out)  #passing it through a linear layer 
        #here we get multiple context rich representations of the token, and we concatenate
        #them along the last dimension, so as to get the final context rich representation, 
        #its a bit different than the original transformer, where the context rich representation
        #is added to the token vector, here we concatenate them, but the idea is the same



#defining the MLP part of the transformer
class FeedForward(nn.Module):
    '''a simple linear layer followed by ReLU'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)  #x is of (B, T, C) shape, and the MLP is applied to each
        #of the token vectors, so as to get the final context rich representation of the token
        #, where C is n_embd



#defining a block, which includes a attention block and a feedforward/MLP block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        #n_embd is the embedding space dimension, n_head is the number of heads of
        #the attention block in a single cumulative block
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)  #just normalises the tokens in the embedding space
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  #passing x through the self attention block, the x + part
        #indicates the residual connection, also utilising the pre layer norm approach here
        x = x + self.ffwd(self.ln2(x))  #passing x through the MLP block, the x + part
        #indicates the residual connection
        return x   #this block will now be repeated multiple times so that 
        #the whole of the transformer is able to communicate with each other and 
        #also derive new info from the context space as much as possible(given already trained)


#defining the transformer
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  #also defining
        #the position encoding for a char based on where it occurs in the context block
        #and its embedded to the same dims as token embedding space, and that is of n_embd

        self.blocks = nn.Sequential(*[Block(n_embd, 4) for _ in range(n_block_layers)])

        self.ln_f = nn.LayerNorm(n_embd)  #also adding a layer norm at the very end of series of blocks
            #within a transformer

        self.lm_head = nn.Linear(n_embd, vocab_size)  #this class hence essentially defines
        #the MLP to convert to logits

    def forward(self, idx, targets=None):
        #idx is (B, T) and targets is also (B, T)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) 
        self.B, self.T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(self.T, device=device))

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  #this layer takes in x as input, (B, T, vocab_size)
        self.B, self.T, self.C = logits.shape
        logits = logits.view(self.B*self.T, self.C)
        
        if targets is None:
            loss = None
        else:
            targets = targets.view(self.B*self.T)
            loss = F.cross_entropy(logits, targets) #but for loss, it want it in form (B, C, T)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_conditioned = idx[:, -block_size:]  # (B, T), so as to ensure
            #that the block_size is maintained as we keep on adding the new tokens, here
            #we take the last block_size tokens of the idx tensor
            logits, loss = self(idx_conditioned)  #calls forward method
            logits = logits.view(self.B, self.T, self.C)[:, -1, :]  #(B, C)
            probs = F.softmax(logits, dim=-1)  #(B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            #append the sampled char to the idx
            idx = torch.cat([idx, idx_next], dim=1)  #(B, T+1)
        return idx


model = BigramLanguageModel()
m =  model.to(device) 


@torch.no_grad()  #the point of this decorator is to turn off the gradient computation
def estimate_loss():  #introduced as the train loss that is printed during the training is 
    #very noisy, to get some stable measuremenat
    out = {}
    model.eval()  #set the model to evaluation mode, by that we mean setting off certain 
    #layer or altering the behaviour of certain layers, for egs, in eval mode the dropout
    #layers are turned off and the batchnorm layers are set to eval mode by using the running
    #mean and var which are recorded during the course of training
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()  #once done with the estimate_loss() func call, set the model back to 
    #training mode, so that the training can continue
    return out
    

#training this model
optimiser = torch.optim.AdamW(m.parameters(), lr=1e-3) 

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()  #to get a stable measure of the loss
        print(f"iter: {iter}, train_loss: {losses['train']}, val_loss: {losses['val']}")

    #get a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)

    #backprop
    optimiser.zero_grad(set_to_none=True)
    loss.backward()

    #update
    optimiser.step()


#generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 500)[0].tolist()))