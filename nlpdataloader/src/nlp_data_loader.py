import torchtext
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom dataset.
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Collate function with batch_first=False (default).
def collate_fn_bfFALSE(batch):
    # Tokenize each sample in the batch using the specified tokenizer.
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indics and create tensor.
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths.
    # Without specifying a value for batch_first it goes default to False, ensuring that the tensors have shape (max_sequence_length ,batch_size)
    padded_batch = pad_sequence(tensor_batch)
    
    # Return padded batch.
    return padded_batch

def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer.
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample.
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence.
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length).
    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    
    # Return the padded batch.
    return padded_batch

# Tokenizer.
tokenizer = get_tokenizer("basic_english")

# Build vocabulary.
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create an instance of your custom data set.
custom_dataset = CustomDataset(sentences)

# Define a batch size.
batch_size = 2

# Data loader with collate_function_bfFALSE.
dataloader_bfFALSE = DataLoader(
    dataset=custom_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn = collate_fn_bfFALSE
)

# Data loader with collate_function.
dataloader = DataLoader(
    dataset=custom_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# Iterate through the data loader with batch_first=True and printing batches
print("Iterate through the data loader with batch_first=True and printing batches")
for batch in dataloader:    
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])

print("---------------------------------------------")

# Iterate through the data loader with batch_first=True and printing vocab
print("Iterate through the data loader with batch_first=True and printing vocab")
for batch in dataloader:
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)

print("---------------------------------------------")

# Iterate through the data loader with batch_first=False and printing batches
print("Iterate through the data loader with batch_first=False and printing batches")
for batch in dataloader_bfFALSE:    
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])

print("---------------------------------------------")

# Iterate through the data loader with batch_first=False and printing vocab
print("Iterate through the data loader with batch_first=False and printing vocab")
for batch in dataloader_bfFALSE:
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)

print("---------------------------------------------")

print("Note that as we made all transformations inside the collate_function, the original dataset remains untouched.")
print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")