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

# run generation
out_basedir = "circuits"
attn_imagedir = out_basedir + "/attention_images"
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
        print('---------------END')