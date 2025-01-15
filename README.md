# This is an implementation of a small version of GPT1(with a little under 10M paramters) and GPT2(124M params) from Andrej Karpathy's zero to hero series

> ## The transformer model architecture
> ![nanoGPT](image-1.png)

### some other vidoes which helped me understand transformers and the mechanism behind attention:

> ## General overview of transformers
> [![3b1b](https://yt3.googleusercontent.com/ytc/AIdro_nFzZFPLxPZRHcE3SSwzdrbuWqfoWYwLAu0_2iO6blQYAU=s160-c-k-c0x00ffffff-no-rj)](https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6)

> ## Attention is only 1/3rd of what you need :)
> [![3b1b](https://yt3.googleusercontent.com/ytc/AIdro_nFzZFPLxPZRHcE3SSwzdrbuWqfoWYwLAu0_2iO6blQYAU=s160-c-k-c0x00ffffff-no-rj)](https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=7)

> ## MLPs within a transformer model
> [![3b1b](https://yt3.googleusercontent.com/ytc/AIdro_nFzZFPLxPZRHcE3SSwzdrbuWqfoWYwLAu0_2iO6blQYAU=s160-c-k-c0x00ffffff-no-rj)](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8)

### Chris Olah's blogs are also a must read to understand about this and related topics
> ### https://colah.github.io/

### And last but not the least, the zero to hero series by none other than Mr Andrej Karpathy himself
> ### [![ak](image-2.png)](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

 ### I already pretrained the model and saved the models params as a torch state dict, which is present in nanoGPT.pth, here is how u can generate some of your own Shakespearean text 
 ### ``` python3 generation.py ```
 ### enter the inputs, and you have new shakespear :)
 ### The upper cap is 5000 words(for no apparent reason)

### an egs i liked(from first iteration of nanoGPT):
![alt text](image.png)

### from a newer one which can take in prompts(starting text)
![alt text](image-3.png)
### pretty good for a "nano" GPT!!

### This will be scaled up to a certain degree, so keep an eye!

### TODO: make a server script that can show the generated shakespear content to the user, given they give
### in the prompt, also have to beautify the README.md page significantly, mention the training process(quite a  story) and add more images(why not)
