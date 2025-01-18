#trained the model on colab, took around an hour to train with its T4 GPU

#using the downloaded model state dict to generate some text

#the code used to download the model state dict
'''
torch.save(model.state_dict(), 'nanoGPT.pth')
from google.colab import files
files.download('nanoGPT.pth')
print("Model saved successfully!")
'''

from src.bigram import BigramLanguageModel
import warnings
import src.bigram as bigram
import torch
import sys

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")  

    bigram.model.load_state_dict(torch.load('nanoGPT.pth', map_location=device)) #move the saved
    #model state dict into the device(mps) 
    bigram.model.to(device)  #then also moved to the device
    bigram.model.eval()


    with torch.no_grad():
        num_generation = input("enter the number of chars to be generated(cap at 5000): ")
        print("before entering any starting promp, please take the following into consideration:")
        print("should only be english letters, with basic punctuation, space")
        print("limited to 256 characters")
        print("in a shakespearean format(roughly)")
        input_context = input("enter the starting prompt (enter 0 for auto generation): ")
        if input_context == '0':
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.tensor([bigram.encode(i) for i in input_context], dtype=torch.long, device=device)
        if num_generation in [str(i) for i in range(5000)]:
            print(input_context)
            print(bigram.decode(bigram.model.generate(context, int(num_generation))[0].tolist()))
        else:
            print('provide how many words to generate')

if __name__ == "__main__":
    main()