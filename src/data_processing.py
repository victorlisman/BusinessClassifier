# src/data_processing.py
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def read_data(companies_path, taxonomy_path):
    companies = pd.read_csv(companies_path)
    taxonomy = pd.read_csv(taxonomy_path)
    return companies, taxonomy

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = list(set(tokens))
    return ' '.join(tokens)

def tokenize(text):
    return clean_text(text).split()

def combine_fields_without_desc(business_tags, sector, category, niche):
    if isinstance(business_tags, list):
        business_tags = " ".join(business_tags)
    combined = f"{business_tags} {sector} {category} {niche}"
    return combined

def combine_fields(description, business_tags, sector, category, niche):
    if isinstance(business_tags, list):
        business_tags = " ".join(business_tags)
    combined = f"{description} {business_tags} {sector} {category} {niche}"
    return combined

def build_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {token: idx+2 for idx, (token, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_to_indices(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]