import torch
import numpy as np

def evaluate_rmse(model, data_loader, device):
    """
    Evaluates the model's RMSE on explicit rating predictions.
    """
    model.eval()
    squared_error = 0.0
    total_ratings = 0
    
    with torch.no_grad():
        for users, items, ratings in data_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            predictions = model(users, items)
            
            # Clip predictions to valid rating range [1.0, 5.0]
            predictions = torch.clamp(predictions, 1.0, 5.0)
            
            squared_error += torch.sum((ratings - predictions) ** 2).item()
            total_ratings += len(ratings)
            
    rmse = np.sqrt(squared_error / total_ratings)
    return rmse

def evaluate_ranking(model, test_instances, device, top_k=10, batch_size=2048):
    """
    Vectorized evaluation of Hit Rate (HR@K) and NDCG@K using leave-one-out.
    test_instances: dict mapping user_idx -> {"pos": pos_item_idx, "negs": list of 99 negs}
    """
    model.eval()
    
    # Prepare tensors for batch evaluation
    user_list = []
    item_list = []
    
    # Keep track of user mapping
    users_sorted = sorted(test_instances.keys())
    
    for u in users_sorted:
        pos_item = test_instances[u]["pos"]
        neg_items = test_instances[u]["negs"]
        
        # We place the positive item at index 0, followed by 99 negative items
        items = [pos_item] + neg_items
        
        user_list.append([u] * len(items))
        item_list.append(items)
        
    user_tensor = torch.tensor(user_list, dtype=torch.long) # Shape: (NumUsers, 100)
    item_tensor = torch.tensor(item_list, dtype=torch.long) # Shape: (NumUsers, 100)
    
    num_users = user_tensor.shape[0]
    num_items_per_user = user_tensor.shape[1]
    
    # Flatten for prediction
    flat_users = user_tensor.view(-1)
    flat_items = item_tensor.view(-1)
    flat_scores = []
    
    # Predict scores in batches
    with torch.no_grad():
        for i in range(0, len(flat_users), batch_size):
            u_batch = flat_users[i : i + batch_size].to(device)
            i_batch = flat_items[i : i + batch_size].to(device)
            scores = model(u_batch, i_batch)
            flat_scores.append(scores.cpu())
            
    # Reshape scores back to (NumUsers, 100)
    scores_matrix = torch.cat(flat_scores).view(num_users, num_items_per_user)
    
    # Top-K ranking
    # The positive item is at column index 0 in the items matrix.
    # We want to check if index 0 ends up in the top-K scores.
    # torch.topk returns values and indices of the top-K elements for each row.
    # Let's get the top-K indices.
    _, topk_indices = torch.topk(scores_matrix, top_k, dim=1) # Shape: (NumUsers, K)
    
    # Since the positive item was at index 0, a HIT is when the index '0' is present in topk_indices
    hits = (topk_indices == 0).any(dim=1).float() # Shape: (NumUsers,)
    hr = torch.mean(hits).item()
    
    # NDCG calculation:
    # If the positive item is hit, calculate its 0-indexed position within the top-K.
    # If the positive item is not hit, NDCG is 0.0.
    ndcg_list = []
    for u_idx in range(num_users):
        row_hits = (topk_indices[u_idx] == 0).nonzero(as_tuple=True)[0]
        if len(row_hits) > 0:
            rank = row_hits[0].item() # 0-indexed rank (0 to K-1)
            ndcg = 1.0 / np.log2(rank + 2.0)
        else:
            ndcg = 0.0
        ndcg_list.append(ndcg)
        
    ndcg = np.mean(ndcg_list)
    
    return hr, ndcg
