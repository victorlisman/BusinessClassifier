# src/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, dataloader, id2label, unique_labels, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = accuracy_score(all_labels, all_preds)
    unique_test_labels = np.union1d(all_labels, all_preds)
    target_names = [id2label[i] for i in unique_test_labels]
    report = classification_report(all_labels, all_preds, labels=unique_test_labels, target_names=target_names, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Accuracy: {:.4f}".format(acc))
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
