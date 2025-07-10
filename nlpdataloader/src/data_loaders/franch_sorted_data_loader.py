import torchtext
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
import pandas as pd
import random
import spacy

sentences = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]

# Define a custom dataset.
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Collate function with batch_first=True.
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
spacy_fr = spacy.load("fr_core_news_sm")
tokenizer = get_tokenizer(lambda x: [tok.text for tok in spacy_fr(x)])

# Build vocabulary.
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Sorts the sentences by length.
sorted_sentences = sorted(sentences, key=lambda s: len(tokenizer(s)))

# Create an instance of your custom data set.
sorted_dataset = CustomDataset(sorted_sentences)

# Define a batch size.
batch_size = 4

# Data loader with collate_function.
dataloader = DataLoader(
    dataset=sorted_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

print("Printing sorted dataset.")
for data in sorted_dataset:
    print(data)

print("---------------------------------------------")

print("Printing batches.")
for batch in dataloader:
    print(batch)