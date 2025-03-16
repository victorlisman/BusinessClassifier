# src/training.py
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np

def train_model(model, dataloader, test_dataloader, criterion, optimizer, device, num_epochs=40):
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for texts, labels, lengths in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
        
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for texts, labels, lengths in test_dataloader:
                texts = texts.to(device)
                labels = labels.to(device)
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        avg_test_loss = test_loss / len(test_dataloader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        test_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Training complete.")
    return model
