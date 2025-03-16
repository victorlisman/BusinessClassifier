# src/labeling.py
import torch
import torch.nn.functional as F
from src.data_processing import clean_text, combine_fields
from src.data_processing import text_to_indices

def predict_labels_for_extra(companies_extra, model, vocab, id2label, device='cuda'):
    companies_extra['combined_text'] = companies_extra.apply(
        lambda row: combine_fields(row['description'], row['business_tags'], row['sector'], row['category'], row['niche']),
        axis=1
    )
    companies_extra['combined_text'] = companies_extra['combined_text'].apply(clean_text)
    pseudo_labels = []
    confidence_scores = []
    model.eval()
    for idx, row in companies_extra.iterrows():
        text = row['combined_text']
        indices = text_to_indices(text, vocab)
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        lengths = [len(indices)]
        with torch.no_grad():
            output = model(input_tensor, lengths)
            probs = F.softmax(output, dim=1)
            max_prob, pred_class = torch.max(probs, dim=1)
        pseudo_labels.append(pred_class.item())
        confidence_scores.append(max_prob.item())
    companies_extra['labels'] = [id2label[idx] for idx in pseudo_labels]
    companies_extra['confidence'] = confidence_scores
    return companies_extra
