"""
Prepare the TinyStories dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import os
import numpy as np
import pickle
import multiprocessing

from datasets import load_dataset # huggingface datasets
from tqdm import tqdm

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = multiprocessing.cpu_count() // 2

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# takes 17GB in huggingface .cache dir, about 30.7M documents (30,720,769)
dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)

def clean(record):
    # only ascii chars
    record['text'] = record['text'].encode('ascii', 'ignore')
    return  record

cleaned_dataset = dataset.map(
    clean,
    desc="cleaning input",
    num_proc=num_proc
)

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

# get all the unique characters that occur in this text as well as total length for training data
desc = "Enumerate characters in training set"
for record in tqdm(cleaned_dataset['train'], desc):
    for char in record['text']:
        chars_dataset.add(char)

# get all the unique characters that occur in this text as well as total length for validation data
desc = "Enumerate characters in validation set"
for record in tqdm(cleaned_dataset['validation'], desc):
    for char in record['text']:
        chars_dataset.add(char)
    
sorted_chars_dataset = sorted(list(chars_dataset))
vocab_size = len(sorted_chars_dataset)

print("all the unique characters:", ''.join(sorted_chars_dataset))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars_dataset) }
itos = { i:ch for i,ch in enumerate(chars_dataset) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

def process(record):
    record['tokens'] = encode(f"{record['text']} ")
    record['len'] = len(record['tokens'])
    return record

tokenized = cleaned_dataset.map(
    process,
    desc="tokenizing the splits",
    num_proc=num_proc
)

# concatenate all the ids in each dataset into one large file we can use for training
len_dataset = 0
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    len_dataset += arr_len
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['tokens'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

print(f"length of dataset in characters: {len_dataset:,}")
