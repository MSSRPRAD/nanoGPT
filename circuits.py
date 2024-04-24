"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "FILE:prompt.txt" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
device = 'cpu'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    print("reading from file")
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# -----------------------------------------------------------------------------
def save_attentions(attentions, output_dir, x_labels, y_labels):
    # return
    num_layers = len(attentions)
    num_heads = attentions[0].size(1)  # Assuming all layers have the same number of heads
    num_tokens = attentions[0].size(-1)
    # Reshape attentions to be more easily visualized

    # Save attention heatmaps
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            plt.imshow(attentions[layer_idx][0][head_idx], cmap='hot', interpolation='nearest')
            plt.xlabel("To")
            plt.ylabel("From")
            plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_head_{head_idx}.png"))
            plt.close()
# -----------------------------------------------------------------------------
import itertools
import seaborn as sns
def argand(a, output_dir, layer_idx, head_idx):
    # print(a.shape)
    # a = a.tolist()
    # print(a)
    # input()

    # Calculate log magnitudes and angles
    log_magnitudes = [np.log(np.abs(z)) for z in a]
    # log_magnitudes = [np.log(z) for z in log_magnitudes]
    max_log_magnitude = max(log_magnitudes)
    log_magnitudes /= max_log_magnitude
    angles = [np.angle(z) for z in a]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.set_xlim(-1.1, 1.1)  # Set x-axis limits
    ax.set_ylim(-1.1, 1.1)  # Set y-axis limits
    ax.spines['left'].set_position('center')  # Move left spine to center
    ax.spines['bottom'].set_position('center')  # Move bottom spine to center
    ax.spines['right'].set_visible(False)  # Hide right spine
    ax.spines['top'].set_visible(False)  # Hide top spine
    ax.set_xlabel('Real', fontsize=5)  # Add x-axis label
    ax.set_ylabel('Imaginary', fontsize=5)  # Add y-axis label
    
    # Plot the points
    colors = sns.color_palette("hls", len(a))  # Get a colormap
    for root, c, log_mag, angle in zip(a, colors, log_magnitudes, angles):
        normalized_log_mag = log_mag  # Convert log magnitude back to normal magnitude
        ax.plot(normalized_log_mag * np.cos(angle), normalized_log_mag * np.sin(angle), marker='x', color=c, markersize=10, alpha=0.8)  # Plot with circles
    
    # Add a circle at the unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)  # Plot the unit circle
    
    # Save the figure
    plt.tight_layout()  # Adjust spacing
    plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_head_{head_idx}.png"), dpi=300, bbox_inches='tight')  # Save with higher resolution
    plt.close()

# -----------------------------------------------------------------------------

def generate_histogram(data, bin_width=0.25, filename="eigenvalue_histograms"):

    num_bins = 8

    bins = np.arange(-1, 1.25, bin_width)

    # Create histogram
    plt.hist(data, bins=bins, stacked=True)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Eigenvalues (sigma eig / sigma abs(eig))')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xticks(np.arange(-1, 1.25, bin_width))
    plt.yticks(np.arange(0, len(data), 1))
    plt.savefig(filename, dpi=300)

# -----------------------------------------------------------------------------

# run generation
out_basedir = "circuits"
attn_imagedir = out_basedir + "/attention_images"
eigenvals_dir = out_basedir + "/eigenvalues"
os.makedirs(out_basedir, exist_ok=True)
os.makedirs(attn_imagedir, exist_ok = True)
with torch.no_grad():
    with ctx:
        print('---------------Context')
        model.circuit = True
        for block in model.transformer.h:
            block.attn.circuit = True
        y = model.circuits(x, max_new_tokens, temperature=temperature, top_k=top_k)
        attns = []
        for block in model.transformer.h:
            attns.append(block.attn.attentions)
        labels = [decode([i]) for i in start_ids]
        save_attentions(attns, attn_imagedir, labels, labels)
        print(decode(x[0].tolist()))
        print('---------------Predicted Next Token')
        print(decode(y.tolist()[0]))
        print('---------------Predicted Logits')
        print(x[0].shape)
        print(model.logits.shape)
        print('---------------Eigenvalues of OV Circuit')
        W_u = model.lm_head.weight
        W_e = model.transformer.wte.weight
        print(W_u.shape, W_e.shape)
        idx = 0
        print('---------')
        for block in model.transformer.h:
            W_o = block.attn.c_proj.weight
            W_v = block.attn.c_attn_v
            eigsums = []
            foo = 0
            for it in W_v:
                print(foo)
                foo += 1
                W_vh = it.weight
                matrix = W_vh @  W_e @ W_u @ W_o
                # print(matrix.shape)
                # print("reached here")
                eigenvalue = torch.linalg.eigvals(matrix).tolist()
                argand(eigenvalue, eigenvals_dir, 0, foo)
                # print(eigenvalue[0].real, eigenvalue[0])
                # print(eigenvalue[0].imag)
                pos = 0
                neg = 0
                eq = 0     
                for eig in eigenvalue:
                    if eig.imag == 0:
                        if eig.real > 0:
                            pos += 1
                        elif eig.real < 0:
                            neg += 1
                        else:
                            eq += 1
                    # print(eig.real/np.sqrt(eig.real*eig.real + eig.imag*eig.imag), eig.imag/np.sqrt(eig.real*eig.real + eig.imag*eig.imag))
                print("trace:")
                print(torch.trace(matrix))
                eig = eigenvalue[0]
                id = 0
                eig_sum = np.sqrt(eig.real*eig.real + eig.imag*eig.imag)
                # print("eig_sum")
                # print(eig_sum)
                for i in eigenvalue:
                    if id != 0:
                        eig += i
                        eig_sum += np.sqrt(i.real*i.real + i.imag*i.imag)
                    # print("eig_sum")
                    # print(eig_sum)
                    id += 1
                       
                eigsums.append(eig.real/eig_sum)
                print("sum of eigenvalues:")
                print(eig / eig_sum)
                # print("pos|neg|eq")
                # print(pos, neg, eq)
                print('% pos|% neg|% eq')
                print(pos/len(eigenvalue), neg/len(eigenvalue), eq/len(eigenvalue))
                idx += 1
                print('---------')
                # print(W_vh.shape)
                # print(W_o.shape)
                # input()
            generate_histogram(eigsums, 0.25, eigenvals_dir + "/eigenvalue_hist.png")
        print('---------------END')

