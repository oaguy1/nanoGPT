"""
Prepare the TinyStories dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import os
import numpy as np
import pickle
import ray

from datasets import load_dataset # huggingface datasets
from tqdm import tqdm

ray.init()

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# takes 17GB in huggingface .cache dir, about 30.7M documents (30,720,769)
dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)

# this results in:
# >>> dataset
# DatasetDict({
#    train: Dataset({
#        features: ['text'],
#        num_rows: 30415547
#    })
#    validation: Dataset({
#        features: ['text'],
#        num_rows: 305105
#    })
#})

chars_dataset = set([])
len_dataset = 0

# get all the unique characters that occur in this text as well as total length for training data
desc = "Enumerate characters in training set"
for story in tqdm(dataset['train']['text'], desc):
    chars = list(set(story))

    for char in chars:
        chars_dataset.add(char)
    
    len_dataset += len(story)

# get all the unique characters that occur in this text as well as total length for validation data
desc = "Enumerate characters in validation set"
for story in tqdm(dataset['validation']['text'], desc):
    chars = list(set(story))

    for char in chars:
        chars_dataset.add(char)
    
    len_dataset += len(story)

sorted_chars_dataset = sorted(list(chars_dataset))
vocab_size = len(sorted_chars_dataset)

print(f"length of dataset in characters: {len_dataset:,}")
print("all the unique characters:", ''.join(sorted_chars_dataset))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars_dataset) }
itos = { i:ch for i,ch in enumerate(chars_dataset) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

@ray.remote
def async_concat(a, b):
    if isinstance(a, str):
        a = np.array(encode(a), dtype=np.uint16)
    if isinstance(b, str):
        b = np.array(encode(b), dtype=np.uint16)
    return np.concatenate((a, b))

# create the train and test splits based on stories for convenience

# iterate over dataset stories, encode story, add it to the correct split
desc = "Encode Training Set"
train_ids = np.array([], dtype=np.uint16)
chunk_size = 50000
num_chunks = len(dataset['train']['text']) // chunk_size
for i in range(num_chunks):
    print(f"{desc}: Chunk {i}/{num_chunks} ({i/num_chunks:.2%})", end="\r")
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size

    if end_idx > len(dataset['train']['text']):
        chunk = dataset['train']['text'][start_idx:]
    else:
        chunk = dataset['train']['text'][start_idx:end_idx]

    while len(chunk) > 1:
        chunk = chunk[2:] + [async_concat.remote(chunk[0], chunk[1])]
    curr_train_ids = ray.get(chunk[0])
    train_ids = np.concatenate((train_ids, curr_train_ids))

print(f"train has {len(train_ids):,} tokens")
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
 
val_encoded = []
for story in tqdm(dataset['validation']['text'], desc): 
    # determin if this is in the val or val set
    val_encoded.append(encode.remote(story))
val_ids = np.array(ray.get(val_encoded), dtype=np.uint16)

print(f"val has {len(val_ids):,} tokens")
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
# export to bin files

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 4,160,818,075
# all the unique characters:       !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz{|}~ ¡¢£§«­°´·»¿ÂÉßàáâåèéêíïñóöúāİœɪʏʙʜіғᴀᴄᴅᴇᴏᴛᴜᴡᴢ   ​‌‎‐‑‒–—―‘’‚“”„…‪′€™−─❤　。」一了些他但保個們兒兩分到剛又和在天奮她己巴度很恩應把整是時會獨玉田留當的童答米給自興艾莉裡這過難高ﬁﬂ﻿，￼�𝑐🌴🌹🍌🍞🎓💖🙂🤩
# vocab size: 242
