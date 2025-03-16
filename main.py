# main.py
import os
import pandas as pd
from src.data_processing import read_data, clean_text, build_vocab, combine_fields, text_to_indices
from src.filtering import filter_companies
from src.labeling import predict_labels_for_extra
from src.training import train_model
from src.evaluation import evaluate_model
from models.lstm_classifier import LSTMClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Set paths
companies_path = os.path.join("data", "raw", "ml_insurance_challenge.csv")
taxonomy_path = os.path.join("data", "raw", "insurance_taxonomy.csv")

companies, taxonomy = read_data(companies_path, taxonomy_path)

# Filter companies using noise filtering
companies_filtered, companies_extra, flagged_indexes = filter_companies(companies, threshold=0.45, device='cuda')
print("Filtered companies:", len(companies_filtered))
print("Extra companies:", len(companies_extra))

companies_filtered.to_csv(os.path.join("data", "processed", "filtered_companies.csv"), index=False)

companies_filtered = companies_filtered.reset_index(drop=True)

companies_filtered['combined_text'] = companies_filtered.apply(lambda row: combine_fields(
    row['description'], row['business_tags'], row['sector'], row['category'], row['niche']
), axis=1)
companies_filtered['combined_text'] = companies_filtered['combined_text'].apply(clean_text)

vocab = build_vocab(companies_filtered['combined_text'].tolist(), min_freq=1)

# Create label mappings
unique_labels = list(set([label for sublist in companies_filtered['labels'].tolist() for label in sublist]))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2idx.items()}

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("label2idx.pkl", "wb") as f:
    pickle.dump(label2idx, f)

train_df, temp_df = train_test_split(companies_filtered, test_size=20, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

from src.data_processing import tokenize 
from torch.utils.data import DataLoader, Dataset
from src.utils import synonym_replacement

class CompanyDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2idx):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2idx = label2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = text_to_indices(text, self.vocab)
        import random
        if random.random() < 0.5:
            text = synonym_replacement(text, n=1)
        label = self.label2idx[self.labels[idx][0]]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_len = max(lengths)
    padded_texts = [torch.cat([text, torch.tensor([vocab['<PAD>']] * (max_len - len(text)), dtype=torch.long)]) for text in texts]
    padded_texts = torch.stack(padded_texts)
    labels = torch.stack(labels)
    return padded_texts, labels, lengths

train_dataset = CompanyDataset(train_df['combined_text'].tolist(), train_df['labels'].tolist(), vocab, label2idx)
test_dataset = CompanyDataset(test_df['combined_text'].tolist(), test_df['labels'].tolist(), vocab, label2idx)
val_dataset = CompanyDataset(val_df['combined_text'].tolist(), val_df['labels'].tolist(), vocab, label2idx)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = len(unique_labels)
num_layers = 1
dropout = 0.5

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, vocab=vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

model = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs=40)

from src.evaluation import evaluate_model
evaluate_model(model, val_dataloader, id2label, unique_labels, device)

# Predict labels for companies_extra (the previously filtered-out companies)
from src.labeling import predict_labels_for_extra
companies_extra = companies_extra.reset_index(drop=True)
companies_extra = predict_labels_for_extra(companies_extra, model, vocab, id2label, device=device)

combined_df = pd.concat([companies_filtered, companies_extra], ignore_index=True)
combined_df.to_csv(os.path.join("data", "processed", "all_companies_tagged.csv"), index=False)

print("Combined DataFrame shape:", combined_df.shape)
