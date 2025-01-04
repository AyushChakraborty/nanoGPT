#trained the model on colab, took around an hour to train with its T4 GPU

#using the downloaded model state dict to generate some text

#the code used to download the model state dict
'''
torch.save(model.state_dict(), 'nanoGPT.pth')
from google.colab import files
files.download('nanoGPT.pth')
print("Model saved successfully!")
'''

from bigram import BigramLanguageModel
import warnings
import bigram
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("mps" if torch.mps.is_available() else "cpu")  

bigram.model.load_state_dict(torch.load('nanoGPT.pth', map_location=device)) #move the saved
#model state dict into the device(mps) 
bigram.model.to(device)  #then also moved to the device
bigram.model.eval()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    print(bigram.decode(bigram.model.generate(context, 500)[0].tolist()))

