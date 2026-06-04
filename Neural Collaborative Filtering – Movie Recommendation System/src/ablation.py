import os
import pandas as pd
import torch
from src.data_utils import get_dataloaders
from src.models import NCFModel
from src.train import train_model

def run_embedding_ablation(train_df, test_df, num_users, num_movies, device, epochs=5, quick_mode=False):
    """
    Varies embedding size (latent dimension) for NCF and evaluates HR@10 and NDCG@10.
    """
    print("\n" + "="*50)
    print("Running Embedding Size Ablation Study...")
    print("="*50)
    
    embedding_sizes = [8, 16, 32, 64] if not quick_mode else [8, 16]
    results = []
    
    # We fix the negative ratio to 4 and MLP depth to 3 layers
    neg_ratio = 4
    
    for size in embedding_sizes:
        print(f"\n[Ablation] Testing Embedding Size: {size}")
        # Build layer dimensions: [2*size, size, size//2]
        mlp_layers = [size * 2, size, size // 2]
        
        train_loader, test_instances = get_dataloaders(
            train_df, test_df, num_users, num_movies, 
            mode="implicit", batch_size=512 if quick_mode else 1024, num_negatives=neg_ratio
        )
        
        model = NCFModel(
            num_users=num_users, 
            num_items=num_movies, 
            latent_dim_gmf=size, 
            latent_dim_mlp=size, 
            mlp_layers=mlp_layers,
            dropout=0.2
        )
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_data=test_instances,
            num_users=num_users,
            num_movies=num_movies,
            mode="implicit",
            epochs=epochs,
            lr=0.002,
            device=device,
            patience=2,
            model_save_path=None # Do not save weights for ablation
        )
        
        best_hr = max(history["val_metric"])
        best_idx = history["val_metric"].index(best_hr)
        best_ndcg = history["val_ndcg"][best_idx]
        
        results.append({
            "embedding_size": size,
            "hr_at_10": best_hr,
            "ndcg_at_10": best_ndcg
        })
        print(f"[Ablation] Finished Embedding Size: {size} | Best HR@10: {best_hr:.4f}")
        
    return pd.DataFrame(results)

def run_depth_ablation(train_df, test_df, num_users, num_movies, device, epochs=5, quick_mode=False):
    """
    Varies MLP layer depth for NCF and evaluates HR@10 and NDCG@10.
    """
    print("\n" + "="*50)
    print("Running MLP Depth Ablation Study...")
    print("="*50)
    
    # MLP configurations with different depths. GMF/MLP embedding size is fixed to 16.
    # Latent dim = 16, so the concatenated input to MLP is 32.
    configs = {
        "Depth 1": [32, 16],
        "Depth 2": [32, 16, 8],
        "Depth 3": [32, 16, 8, 4]
    }
    
    if quick_mode:
        configs = {
            "Depth 1": [32, 16],
            "Depth 2": [32, 16, 8]
        }
        
    results = []
    size = 16
    neg_ratio = 4
    
    train_loader, test_instances = get_dataloaders(
        train_df, test_df, num_users, num_movies, 
        mode="implicit", batch_size=512 if quick_mode else 1024, num_negatives=neg_ratio
    )
    
    for depth_name, layers in configs.items():
        print(f"\n[Ablation] Testing MLP Layers: {layers} ({depth_name})")
        
        model = NCFModel(
            num_users=num_users, 
            num_items=num_movies, 
            latent_dim_gmf=size, 
            latent_dim_mlp=size, 
            mlp_layers=layers,
            dropout=0.2
        )
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_data=test_instances,
            num_users=num_users,
            num_movies=num_movies,
            mode="implicit",
            epochs=epochs,
            lr=0.002,
            device=device,
            patience=2,
            model_save_path=None
        )
        
        best_hr = max(history["val_metric"])
        best_idx = history["val_metric"].index(best_hr)
        best_ndcg = history["val_ndcg"][best_idx]
        
        results.append({
            "depth_name": depth_name,
            "num_layers": len(layers) - 1, # Number of hidden layers
            "hr_at_10": best_hr,
            "ndcg_at_10": best_ndcg
        })
        print(f"[Ablation] Finished {depth_name} | Best HR@10: {best_hr:.4f}")
        
    return pd.DataFrame(results)

def run_neg_ratio_ablation(train_df, test_df, num_users, num_movies, device, epochs=5, quick_mode=False):
    """
    Varies negative sampling ratio for NCF and evaluates HR@10 and NDCG@10.
    """
    print("\n" + "="*50)
    print("Running Negative Sampling Ratio Ablation Study...")
    print("="*50)
    
    neg_ratios = [1, 2, 4, 6] if not quick_mode else [1, 3]
    results = []
    
    # Fix embedding size to 16, MLP layers to [32, 16, 8]
    size = 16
    mlp_layers = [32, 16, 8]
    
    for neg_ratio in neg_ratios:
        print(f"\n[Ablation] Testing Negative Sampling Ratio: {neg_ratio}")
        
        train_loader, test_instances = get_dataloaders(
            train_df, test_df, num_users, num_movies, 
            mode="implicit", batch_size=512 if quick_mode else 1024, num_negatives=neg_ratio
        )
        
        model = NCFModel(
            num_users=num_users, 
            num_items=num_movies, 
            latent_dim_gmf=size, 
            latent_dim_mlp=size, 
            mlp_layers=mlp_layers,
            dropout=0.2
        )
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_data=test_instances,
            num_users=num_users,
            num_movies=num_movies,
            mode="implicit",
            epochs=epochs,
            lr=0.002,
            device=device,
            patience=2,
            model_save_path=None
        )
        
        best_hr = max(history["val_metric"])
        best_idx = history["val_metric"].index(best_hr)
        best_ndcg = history["val_ndcg"][best_idx]
        
        results.append({
            "neg_ratio": neg_ratio,
            "hr_at_10": best_hr,
            "ndcg_at_10": best_ndcg
        })
        print(f"[Ablation] Finished Neg Ratio: {neg_ratio} | Best HR@10: {best_hr:.4f}")
        
    return pd.DataFrame(results)

def run_all_ablation_studies(train_df, test_df, num_users, num_movies, device, results_dir="results", epochs=5, quick_mode=False):
    """
    Runs all ablation studies and saves results to CSV.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Run embedding size ablation
    df_embed = run_embedding_ablation(train_df, test_df, num_users, num_movies, device, epochs, quick_mode)
    df_embed.to_csv(os.path.join(results_dir, "ablation_embedding.csv"), index=False)
    
    # Run MLP depth ablation
    df_depth = run_depth_ablation(train_df, test_df, num_users, num_movies, device, epochs, quick_mode)
    df_depth.to_csv(os.path.join(results_dir, "ablation_depth.csv"), index=False)
    
    # Run negative sampling ratio ablation
    df_neg = run_neg_ratio_ablation(train_df, test_df, num_users, num_movies, device, epochs, quick_mode)
    df_neg.to_csv(os.path.join(results_dir, "ablation_neg_ratio.csv"), index=False)
    
    print("\n[Ablation] All ablation studies finished and results saved in results/ folder.")
