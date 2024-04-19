import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    data_urls = [
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Sorcerer's%20Stone.txt",
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Chamber%20of%20Secrets.txt",
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20The%20Half-Blood%20Prince.txt",
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Deathly%20Hallows%20.txt",
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Goblet%20of%20Fire.txt",
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Order%20of%20the%20Phoenix.txt",
        "https://raw.githubusercontent.com/bobdeng/owlreader/master/ERead/assets/books/Harry%20Potter%20and%20the%20Prisoner%20of%20Azkaban%20.txt",
        "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
        "https://www.gutenberg.org/cache/epub/43/pg43.txt",
        "https://www.gutenberg.org/cache/epub/6130/pg6130.txt",
        "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
        "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
        "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
        "https://www.gutenberg.org/cache/epub/1259/pg1259.txt",
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
        "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
        "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
        "https://www.gutenberg.org/cache/epub/600/pg600.txt",
        "https://www.gutenberg.org/cache/epub/36034/pg36034.txt",
        "https://www.gutenberg.org/cache/epub/2638/pg2638.txt",
        "https://www.gutenberg.org/cache/epub/40745/pg40745.txt",
        "https://www.gutenberg.org/cache/epub/2197/pg2197.txt",
        "https://www.gutenberg.org/cache/epub/8117/pg8117.txt",
        "https://www.gutenberg.org/cache/epub/98/pg98.txt",
        "https://www.gutenberg.org/cache/epub/48371/pg48371.txt"
    ]

    with open(input_file_path, 'a', encoding='utf-8') as f:
        for data_url in tqdm(data_urls, desc="Downloading data"):
            response = requests.get(data_url)
            f.write(response.text)
            f.write("\n\n")
        print("Writing wikics.......")
        wikics_dataset = load_dataset("AlaaElhilo/Wikipedia_ComputerScience")
        for item in wikics_dataset['train']:
            f.write(item['Text'])
            f.write("\n\n")
        print("Writing tinystories.....")
        tinystories_dataset = load_dataset("georgeyw/TinyStoriesV2-GPT4")
        for item in tinystories_dataset['train']:
            f.write(item['text'])
            f.write("\n\n")
        print("Writing Wikidata en descriptions......")
        wikidata_en_desc_dataset = load_dataset("derenrich/wikidata-en-descriptions-small")
        for item in wikidata_en_desc_dataset['train']:
            print(item['input'])
            input()
            f.write(item['input'])
            f.write("\n\n") 
        print("Writing stanford encyclopedia philosophy.....")
        stanford_ency_phil_dataset = load_dataset("AiresPucrs/stanford-encyclopedia-philosophy")
        for item in stanford_ency_phil_dataset['train']:
            f.write(item['text'])
            f.write("\n\n") 
        print("Writing yelp reviews.....")
        yelp_reviews_dataset = load_dataset("yelp_review_full")
        for item in yelp_reviews_dataset['train']:
            f.write(item['text'])
            f.write("\n\n") 
        print("Writing wikitext......")
        wikitext_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        for item in wikitext_dataset['train']:
            f.write(item['text'])
            f.write("\n\n") 
          

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
