from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import uvicorn
import pickle
import numpy as np
import nltk
# uncomment if not downloaded
#nltk.download('punkt')
#nltk.download('stopwords')
from models.lstm_classifier import LSTMClassifier
from src.data_processing import clean_text, text_to_indices, combine_fields

app = FastAPI()



with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("label2idx.pkl", "rb") as f:
    label2idx = pickle.load(f)

id2label = {idx: label for label, idx in label2idx.items()}

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = len(label2idx)
num_layers = 1
dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, vocab=vocab)
model_path = "lstm_classifier.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the request body for prediction.
class PredictionRequest(BaseModel):
    description: str
    business_tags: list  
    sector: str
    category: str
    niche: str

@app.post("/predict")
def predict(request: PredictionRequest):
    combined_text = combine_fields(
        request.description, request.business_tags, request.sector, request.category, request.niche
    )
    cleaned = clean_text(combined_text)
    indices = text_to_indices(cleaned, vocab)
    if not indices:
        return {"error": "Input text is too short or could not be processed."}
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = [len(indices)]
    with torch.no_grad():
        outputs = model(input_tensor, lengths)
        pred_idx = torch.argmax(outputs, dim=1).item()
    pred_label = id2label[pred_idx]
    return {"predicted_label": pred_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
