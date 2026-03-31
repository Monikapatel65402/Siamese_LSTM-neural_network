import os
import pickle
import collections
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()
MAX_LEN = int(os.getenv("MAX_LEN", 12))
MAX_VOCAB = 15000

# Re-declare SimpleTokenizer to match pickle loading
class SimpleTokenizer:
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 1
        
    def fit_on_texts(self, texts):
        counter = collections.Counter()
        for text in texts:
            words = str(text).lower().split()
            counter.update(words)
            
        most_common = counter.most_common(self.num_words - 1 if self.num_words else None)
        for idx, (word, _) in enumerate(most_common, start=1):
            self.word_index[word] = idx
            self.index_word[idx] = word
        self.vocab_size = len(self.word_index) + 1
            
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = str(text).lower().split()
            seq = [self.word_index[w] for w in words if w in self.word_index]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        if length > 0:
            padded[i, -length:] = seq[:length]
    return padded

# Re-declare PyTorch SiameseNetwork
class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, max_len=12):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm1 = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(64)
        
        self.fc1 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward_once(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (hn, _) = self.lstm2(x)
        x = hn[-1] # take last hidden state
        x = self.batch_norm(x)
        return x
        
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        
        # Manhattan distance
        distance = torch.abs(out1 - out2)
        
        x = torch.relu(self.fc1(distance))
        x = self.dropout2(x)
        x = self.sigmoid(self.out(x))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
sys.modules['__main__'].SimpleTokenizer = SimpleTokenizer

print("Loading tokenizer...")
tokenizer_path = os.getenv("TOKENIZER_PATH", "../model/tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

print("Loading model...")
model_path = os.getenv("MODEL_PATH", "../model/best_model.pt")
model = SiameseNetwork(MAX_VOCAB, 128, MAX_LEN).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Ready!")

def get_recommendations(product_name: str, catalog: list, top_n: int = 5):
    main_seq = pad_sequences(tokenizer.texts_to_sequences([product_name]), maxlen=MAX_LEN)
    catalog_seq = pad_sequences(tokenizer.texts_to_sequences(catalog), maxlen=MAX_LEN)
    main_repeated = np.repeat(main_seq, len(catalog), axis=0)
    
    with torch.no_grad():
        x1 = torch.tensor(main_repeated, dtype=torch.long).to(device)
        x2 = torch.tensor(catalog_seq, dtype=torch.long).to(device)
        scores = model(x1, x2).squeeze().cpu().numpy()
        
    # If the catalog only has 1 item, scores is a scalar not an array
    if scores.ndim == 0:
        scores = np.array([scores])
        
    top_idx = scores.argsort()[-top_n:][::-1]
    return [
        {"product": catalog[i], "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]
