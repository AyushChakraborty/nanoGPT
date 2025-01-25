import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))   #add parent dir to sys.path

from GPTs import bigram
import warnings
import torch

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")  

    bigram.model.load_state_dict(torch.load('../saved_models/nanoGPT.pth', map_location=device)) #move the saved
    bigram.model.to(device)
    bigram.model.eval()


    with torch.no_grad():
        if len(sys.argv) == 1:
            print("no prompts given")
            exit(1)

        num_generation = sys.argv[1]    #number of chars to generate
        input_context = sys.argv[2]     #a string which comes in
        if input_context == '0':
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.tensor([bigram.encode(i) for i in input_context], dtype=torch.long, device=device)
        if num_generation in [str(i) for i in range(5000)]:
            if input_context != '0':
                print(input_context)
            bigram.decode(bigram.model.generate(context, int(num_generation))[0].tolist())
        else:
            print('provide how many words to generate')

if __name__ == "__main__":
    main()