# src/filtering.py
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from src.data_processing import clean_text, combine_fields_without_desc

def get_model(device='cuda'):
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

def filter_companies(companies, threshold=0.45, device='cuda'):
    model = get_model(device)
    flagged_indexes = []
    for idx, row in companies.iterrows():
        description = str(row['description'])
        business_tags = row['business_tags']
        sector = str(row['sector'])
        category = str(row['category'])
        niche = str(row['niche'])
        combined_text = combine_fields_without_desc(business_tags, sector, category, niche)
        cleaned_description = clean_text(description)
        cleaned_combined = clean_text(combined_text)
        with torch.no_grad():
            desc_embedding = model.encode(cleaned_description, convert_to_tensor=True)
            combined_embedding = model.encode(cleaned_combined, convert_to_tensor=True)
        cos_sim = F.cosine_similarity(desc_embedding, combined_embedding, dim=0).item()
        if cos_sim < threshold:
            flagged_indexes.append(idx)
    companies_filtered = companies.drop(flagged_indexes)
    remaining_idx = companies.index.difference(companies_filtered.index)
    companies_extra = companies.loc[remaining_idx].copy()
    return companies_filtered, companies_extra, flagged_indexes
