from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP    #if DDP ends up getting used, need to wrap the model around it
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from transformers import GPT2LMHeadModel
import tiktoken as tkn
import math
import os
import inspect


#-------------------------
#here the namings are done as per the openai/HF implementation so that the weights can be ported to this 
#implementation of GPT2 easily



if torch.mps.is_available():
    device = torch.device("mps") 
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@dataclass  #this decorator automatically generates an __init__ method for the class with the
            #attributes as variables defined under the class for which the decorator is called
class GPTConfig:
    block_size: int = 1024     #number of tokens in the context window
    vocab_size: int = 50257    #number of unique tokens in the vocabulary, 50000 BPE megres, 256 byte tokens, 1 <|endoftext|> token, matches the GPT2 tokeniser
    n_layer: int = 12          #number of blocks in the model
    n_head: int = 12           #number of heads in the multiheadattention
    n_embd: int = 768



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0                #making sure that the head dimension is a factor of the embedding dimension
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  #linear layer for the attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)   
        self.c_proj.NANOGPT2_SCALE_INIT = 1.0
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
        k = k.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)  #(B, n_head, T, head_size)
        #here, instead of adding the vectors obtained from diff heads, we concatenate them and hence
        #we split the vectors n_head parts, each for one head of the attention block, and each having
        #C//n_head which is head_size as number of elements
        q = q.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  #essentially doing (k dot q )/ sqrt(d_k), (B, n_head, T, T)
        #so the (T, T) attention grid for each head for each batch is obtained

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)         #masking phase, also :T is present as during generation, the context window
        #can vary from 1 to T tokens

        y = att @ v  # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
                    #here we get the weighted value vector for each token for each batch and for each head

        #y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  #flash attention part, runs proprly only on cuda with triton installed, so use accordingly

        y = y.transpose(1, 2).contiguous().view(B, T, C)  #here we concatenate those value vectors for a
        #token obtained from all the diff heads across all the tokens and across all the batches, so now 
        #C has size n_head*head_size 
        y = self.c_proj(y)                                #passing through the linear layer

        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')                  #GELU or gaussian error linear unit is a softer version of RELU, where one of its advantage is that there is always some gradient contribution hence the issue of dead gradients is solved
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT2_SCALE_INIT = 1.0

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
            wte = nn.Embedding(config.vocab_size, config.n_embd),               #making the embedding table of vocab_size, n_embd
            wpe = nn.Embedding(config.block_size, config.n_embd),               #making the positional embedding table of block_size, n_embd
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  #blocks
            ln_f = nn.LayerNorm(config.n_embd)                                  #final layer norm layer
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  #the final linear layer, to get the logits

        #weight-sharing scheme between the embedding layer and the unembedding layer
        self.transformer.wte.weight = self.lm_head.weight  #weights of unembedding layer shared with embedding layer, 
        #their shapes still remain the same, its just that wte points to the same tensor in memory location as lm_head, hence 
        #if one updates, the other updates too, but the way in which they are interpreted later on do become different
        #using this also, we can save memory as we dont have to store the same weights in two different places, in our
        #case 40m out of 124m are saved, which is 30 percent

        self.apply(self.init_weights)                      #calling the .apply() method of nn.Module which applies the init_weights method to all the layers in the model


    def init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT2_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5             #times two as for each block/n_layer, there is one attention and one MLP block
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)  #initialising the weights of all the linear layer
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)                    
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx, targets=None):  #used for generation of new tokens, targets are None by default in case we want to use the pretrained model, and during generation
        #idx has shape (B, T) where T <= config.block_size and its obviously varible in this case
        B, T = idx.shape
        assert T <= self.config.block_size, f"cannot forward sequence of length {T} because GPT2 is limited to block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  #(T), only need to pass this tensor to device as the rest are just modifications of this tensor, and they too will be on the device
        pos_emb = self.transformer.wpe(pos)                            #(T, n_embd)
        tok_emb = self.transformer.wte(idx)                            #(B, T, n_embd)
        x = tok_emb + pos_emb                                          #(B, T, n_embd), the same tokens in embedding space but now infused with 
        #positional information

        #since the goal of this method is to generate, we have to step through the blocks, and thats what we do now
        for b in self.transformer.h:
            x = b(x)
        
        x = self.transformer.ln_f(x)  #(B, T, vocab_size), passing it through the final linear layer so as to obtain the logits from the last token
        logits = self.lm_head(x)      #(B, T, vocab_size), the logits for the next token
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  #flattening the logits to (B*T, vocab_size) and targets to (B*T), and then calculating the cross entropy loss, as it prefers flattened tensors
        
        return logits, loss    
        #also since we feed the generation step also in batches it means that multiple sequences can
        #be generated at once


    def configure_optimisers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}  #only take those params which require grad

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]        #setting it such that only the params with dim >= 2 are decayed
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]      #setting it such that only the params with dim < 2 are not decayed

        optim_params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device                         #will turn to False in case of mps, as fused kernels are meant to take advantage of the tensor cores, which are not present in mps
        print(f"using fused adam: {use_fused}")
        optimiser = torch.optim.AdamW(optim_params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimiser


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
        model = GPT(config)        #own implemented model

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn. bias')]
        #now the .attn.bias is a result of the heirarchical naming of torch nn.modules, since
        #self.atnn = CausalSelfAttention(config) is called within Block, and Block is called within GPT, 
        #so keys are named as transformer.h.0.attn.bias and that number can change based on which block
        #we refer to, but to just get the bias/mask tensor, we can use .attn.bias

        #so this naming is just an artifact of the torch nn.Module heirarchy, and not a part of the model

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.masked_bias')]      #in the HF implementation, the bias is named as masked_bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla linear
        # this means that we have to transpose these weights when we import them
        #assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



class DataLoader:
    def __init__(self, B, T, process_rank, num_processes):                 #B is the intended batch size
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tkn.get_encoding('gpt2')        #loading the encoding of gpt2
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)    #its good to not enforce anything on the GPU within APIs, as the user might want to use the API on CPU
        print(f"loaded {len(self.tokens)} tokens")
        print(f"one epoch: {len(self.tokens)//(B*T)} batches")  #as one epoch is completed when we go through the entire dataset once, so upon going through the entire dataset once we would have gone through these many batches
        self.current_state = self.B * self.T * self.num_processes           #to maintain the state, also with support for multiple cuda processes


    def next_batch(self):
        buf = self.tokens[self.current_state : self.current_state + self.B*self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_state += self.B * self.T * self.num_processes          #go to the next batch
        
        #if out of bounds
        if (self.current_state + self.B*self.T*self.num_processes  + 1) > len(self.tokens):
            self.current_state = self.B * self.T * self.num_processes   #go back to the start of the dataset, for that process
        #this new implementation now ensures that if multiple cuda processes are present, then data can be spread
        #across accordingly, ensuring that each process gets its own data, and the data is not repeated across the processes
        #and also the amt of tokens processed as such is also more(depends on the num_processes)
        return x, y



def save_checkpoints(model, optimiser, step, filename="nanogpt2.pth"):
    """saves the model and the optimiser state dict to a file, so that
    if the training gets interrupted, we can resume from the last checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'step': step
    }
    torch.save(checkpoint, filename)
    print(f"checkpoint saved to {filename}")



def load_checkpoints(filename, model, optimiser):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"loaded model from {filename}")

    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    print(f"loaded optimiser from {filename}")

    step = checkpoint['step']
    return step


#-------------------------
import time
if torch.mps.is_available():
    torch.mps.empty_cache()      #clearing the cache
elif torch.cuda.is_available():
    torch.cuda.empty_cache()


#set up DDP(distributed data parallel) for multi-gpu training if present in CUDA specifically
    #torchrun command here sets up the env vars like RANK, LOCAL_RANK, WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1        #is this a ddp run or not
    if ddp:
        #use of DDP depends on CUDA, so we set the device appropriately according to rank
        assert torch.cuda.is_available(), "torch.distributed only works with CUDA"
        init_process_group(backend='nccl') 
        ddp_rank = int(os.environ['RANK'])               #numbering of those processes/gpus
        ddp_world_size = int(os.environ['WORLD_SIZE'])   #total number of processes running
        ddp_local_rank = int(os.environ['LOCAL_RANK'])   
        device = f"cuda:{ddp_local_rank}"                
        torch.cuda.set_device(device)                    #setting the device to the local rank
        master_process = ddp_rank == 0                   #this process does logging
    else:
        #vanilla, non-DDP run
        ddp_rank = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

def main():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    elif torch.mps.is_available():
        torch.mps.manual_seed(1337)

    #simulated batch size init 
    total_batch_size = 524288                          #2**19  ~ 0.5M tokens, faithful to the GPT2 model
    B = 8
    T = 512                                            #if DDP is available and used, then B*T will happen in each of the ddp_world_size processes
    assert total_batch_size % (B*T*ddp_world_size) == 0, "batch size must divide total batch size across all processes(if available)"
    grad_accum_steps = total_batch_size // (B*T*ddp_world_size)         #will give the number of forward backwards to be done before updating the weights
    if master_process:
        print(f"total desired batch size: {total_batch_size}, grad_accum_steps: {grad_accum_steps}") 


    #get the data batch
    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)   #shld be 16, 1024

    torch.set_float32_matmul_precision("high")         #setting to tf32 instead of fp32, for faster computation

    #model = GPT.from_pretrained('gpt2')               #loading the pretrained model, if needed
    model = GPT(GPTConfig(vocab_size=50304))           #changing the vocab size to be power of 2, making it a nice number, hence making training faster, keep in mind that the other tokens dont end up getting used 
    model.to(device)                                   #this sets up the model 
    model = torch.compile(model)                       #uses JIT compiling to speed up the training
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])#setting up the model for DDP, if DDP is used

    raw_model = model.module if ddp else model        

    #implementing cosine decay of learning rate with linear warmup, this is a learning rate scheduler
    max_lr = 12e-4
    min_lr = max_lr * 0.01
    warmup_steps = 40
    max_steps = 150
    def get_lr(step):
        #linear warmup
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        if step > max_steps:
            return min_lr
        #cosine decay
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


    #optimising/training
    optimiser = raw_model.configure_optimisers(weight_decay = 0.1, learning_rate = 6e-4, device=device)
    resumed_step = 0
    save_interval = 4
    train_loss_log = []
    val_loss_log = []

    #check if checkpoint exists
    checkpoint_file = "./saved_models/nanogpt2.pth"
    if os.path.exists(checkpoint_file):
        try:
            resumed_step = load_checkpoints(checkpoint_file, raw_model, optimiser)
        except Exception as e:
            print(f"error loading checkpoint: {e}")
        print(f"resuming from step {resumed_step}")

        for step in range(resumed_step, max_steps):
            t0 = time.time()
            #---------------
            optimiser.zero_grad()
            #------- inner loop for simulated batch size of 0.5M tokens
            loss_accum = 0                            #if DDP is used, then the loss is summed across all the processes
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)                                    #putting the data on the device
                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   #only now converts certain layers like linear to bf16
                        logits, loss = model(x, y)                                   #this feature of mixed precision training for now is only available on cuda devices 
                else:
                    logits, loss = model(x, y)
                loss /= grad_accum_steps           #to account for the fact that in each inidvidual mini batch, the loss is found and then added over
                #all the other mini batches, but if we were to pass the 0.5M tokens as a singular batch, the loss would have a reduction of mean
                #which means that it is divided by the number of tokens in each batch, now that is already taken care of in the individual batches, but
                #amongst the grad_accum_steps number of batches, it is not, hence we divide by grad_accum_steps, to recover the factor back. So 
                #it all has to do with the fact that losses generally have a reduction of mean and that has to be accounted for
                loss_accum += loss.detach()
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  #syncing the gradients across the processes
                loss.backward()
            if ddp:
                all_reduce(loss_accum, op=ReduceOp.AVG)  #summing the loss across all the processes
            #-------
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                   #clipping the gradients to 1.0
            lr = get_lr(step)
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr
            optimiser.step()
            #---------------
            if torch.cuda.is_available():
                torch.cuda.synchronize()                      #waiting for the GPU to finish the computation, then move to the next line which is calculating the finish time
            elif torch.mps.is_available():
                torch.mps.synchronize()    
            t1 = time.time()
            dt = t1 - t0                            #in s
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size        #across the large batch 
            tokens_per_sec = tokens_processed / dt
            print(f'step: {step} | loss: {loss_accum.item(): .6f} | lr: {lr: .4e} | dt: {dt: .4f}s | norm: {norm: .4f} | tokens_per_sec: {tokens_per_sec: .2f}tok/sec')   #loss also lives on GPU, then .item() moves it to CPU, and converts to a float 
            if step % save_interval == 0 and master_process:    #ensures the only the master process saves the checkpoints
                save_checkpoints(raw_model, optimiser, step)    #saving the checkpoints after each step
            train_loss_log.append(loss_accum.item())            #logging the training loss for each step
                

        if ddp:
            destroy_process_group()                 #cleaning up the process group, as the training is done


#not within the main func, so as to generate new text from another module

#tokenising the input 
model = GPT.from_pretrained('gpt2')
model.to(device)
model.eval()                                                  #putting it in eval, altogether not necessary as we are not using dropout or batchnorm
num_return_sequences = 1
max_length = 150
enc = tkn.get_encoding('gpt2')                                #loads the encoding of gpt2 
tokens = enc.encode("haiku ")
tokens = torch.tensor(tokens, dtype=torch.long)               #(12, ), as seen from the tiktokenizer app, chk it out :)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  #(3, 12), so that we can generate 3 sequences at once
idx = tokens.to(device)                                       #putting it on the device

#generating !
#here idx is (num_return_sequences, T), where T is the number of tokens in the input sequence, and num_return_sequences is number of batches
while idx.size(1) < max_length:
    with torch.no_grad():
        idx = idx[:, -max_length:]                #trimming the input sequence to the max_length
        logits, loss = model(idx)                 #(num_return_sequences, T, vocab_size), where again, T is the number of tokens in the input sequence, and varies
        logits = logits[:, -1, :]                 #(num_return_sequences, vocab_size),  get only the last tokens from each of the batches, as what we essentially have now at this point is a bigram problem at hand
        probs = F.softmax(logits, dim=-1)          #(num_return_sequences, vocab_size), the probabilities of the next token

        #doing top-k sampling which is huggingface default
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)       #(num_return_sequences, 50), the top 50 tokens and their probabilities
        ix = torch.multinomial(topk_probs, num_samples=1)              #(num_return_sequences, 1), the indices of the tokens sampled from the top 50 tokens from each one of the batches
        idxcol = torch.gather(topk_indices, -1, ix)                    #(num_return_sequences, 1), the actual token indices sampled from the top 50 tokens
        print(enc.decode([idxcol[0, 0].item()]), end="", flush=True)   #setting num_return_sequences as 1, only then will this work
        idx = torch.cat((idx, idxcol), dim=1)                          #(num_return_sequences, T+1), the new token indices for each of the batches
        

if __name__ == "__main__":
    main()