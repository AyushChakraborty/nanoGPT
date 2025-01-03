import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
device = torch.device("mps" if torch.mps.is_available() else "cpu")  #from now on, will utilize the GPU
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
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


#defining the model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  #also defining
        #the position encoding for a char based on where it occurs in the context block
        #and its embedded to the same dims as token embedding space, and that is of n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)  #this class hence essentially defines
        #the MLP of the transformer, where for now we just pass it through one layer
        #so here the weights are of dim (n_embd, vocab_size)

    def forward(self, idx, targets=None):
        #idx is (B, T) and targets is also (B, T)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) 
        self.B, self.T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(self.T, device=device))
        x = tok_emb + pos_emb  # (B, T, C)
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
            logits, loss = self(idx)  #calls forward method
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