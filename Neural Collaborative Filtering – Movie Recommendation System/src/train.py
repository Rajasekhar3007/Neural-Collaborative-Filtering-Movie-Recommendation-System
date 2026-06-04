import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.evaluate import evaluate_rmse, evaluate_ranking

def train_one_epoch(model, dataloader, optimizer, criterion, device, mode="explicit"):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    
    # Resample negative items for implicit training
    if mode == "implicit" and hasattr(dataloader.dataset, "sample_negatives"):
        dataloader.dataset.sample_negatives()
        
    for users, items, targets in dataloader:
        users, items, targets = users.to(device), items.to(device), targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(users, items)
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        
    return total_loss / len(dataloader.dataset)

def train_model(model, train_loader, val_data, num_users, num_movies, mode="explicit", 
                epochs=20, lr=0.001, weight_decay=1e-5, device="cpu", patience=5, model_save_path=None):
    """
    Complete model training pipeline with early stopping.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if mode == "explicit":
        criterion = nn.MSELoss()
        best_metric = float("inf") # Lower RMSE is better
        print("[Train] Training SVD/NCF in EXPLICIT mode (Rating Prediction)...")
    else:
        criterion = nn.BCEWithLogitsLoss()
        best_metric = -1.0 # Higher HR@10 is better
        print("[Train] Training SVD/NCF in IMPLICIT mode (Top-10 Recommendation)...")
        
    history = {
        "train_loss": [],
        "val_metric": [],
        "val_ndcg": [] if mode == "implicit" else None
    }
    
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mode)
        history["train_loss"].append(loss)
        
        # Evaluate model on validation/test set
        if mode == "explicit":
            # val_data is val_loader
            val_metric = evaluate_rmse(model, val_data, device)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Val RMSE: {val_metric:.4f}")
            
            # Check if RMSE improved
            if val_metric < best_metric:
                best_metric = val_metric
                epochs_no_improve = 0
                if model_save_path:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                
        else:
            # val_data is test_instances dict
            val_hr, val_ndcg = evaluate_ranking(model, val_data, device, top_k=10)
            val_metric = val_hr
            history["val_ndcg"].append(val_ndcg)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f}")
            
            # Check if Hit Rate @ 10 improved
            if val_hr > best_metric:
                best_metric = val_hr
                epochs_no_improve = 0
                if model_save_path:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
            else:
                epochs_no_improve += 1
                
        history["val_metric"].append(val_metric)
        
        # Early Stopping check
        if epochs_no_improve >= patience:
            print(f"[Train] Early stopping triggered. Training stopped after {epoch} epochs.")
            break
            
    # Load the best model weights if saved
    if model_save_path and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        
    return history
