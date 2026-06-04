import os
import argparse
import pandas as pd
import torch
from src.data_utils import download_and_extract, load_ratings, preprocess_data, split_data_explicit, split_data_implicit, get_dataloaders
from src.models import SVDModel, NCFModel
from src.train import train_model
from src.ablation import run_all_ablation_studies
from src.visualize import plot_training_comparison, plot_ablation_results

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Collaborative Filtering & SVD Movie Recommendation System")
    parser.add_argument("--mode", type=str, default="both", choices=["explicit", "implicit", "both"],
                        help="Task mode: 'explicit' (ratings), 'implicit' (ranking/HR@10), or 'both'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension size")
    parser.add_argument("--neg_ratio", type=int, default=4, help="Negative sampling ratio for implicit training")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run training (cuda or cpu)")
    parser.add_argument("--quick-mode", action="store_true", help="Downsample dataset and run fast (1 epoch) for code validation")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip running ablation studies")
    return parser.parse_args()

def run_explicit_pipeline(df, num_users, num_movies, args):
    """
    Executes SVD vs NCF training and evaluation for rating prediction (explicit feedback).
    """
    print("\n" + "="*70)
    print("RUNNING EXPLICIT FEEDBACK PIPELINE (RATING PREDICTION)")
    print("="*70)
    
    train_df, test_df = split_data_explicit(df, test_ratio=0.2)
    
    train_loader, val_loader = get_dataloaders(
        train_df, test_df, num_users, num_movies,
        mode="explicit", batch_size=args.batch_size
    )
    
    # ---------------- 1. Train SVD Model ----------------
    print("\n--- Training SVD (Matrix Factorization) Model ---")
    svd_model = SVDModel(num_users=num_users, num_items=num_movies, embedding_dim=args.embedding_dim)
    svd_save_path = os.path.join("models", "best_svd_explicit.pth")
    svd_history = train_model(
        model=svd_model,
        train_loader=train_loader,
        val_data=val_loader,
        num_users=num_users,
        num_movies=num_movies,
        mode="explicit",
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        model_save_path=svd_save_path
    )
    
    # ---------------- 2. Train NCF Model ----------------
    print("\n--- Training NCF (NeuMF) Model ---")
    mlp_layers = [args.embedding_dim * 2, args.embedding_dim, args.embedding_dim // 2]
    ncf_model = NCFModel(
        num_users=num_users, 
        num_items=num_movies, 
        latent_dim_gmf=args.embedding_dim, 
        latent_dim_mlp=args.embedding_dim,
        mlp_layers=mlp_layers
    )
    ncf_save_path = os.path.join("models", "best_ncf_explicit.pth")
    ncf_history = train_model(
        model=ncf_model,
        train_loader=train_loader,
        val_data=val_loader,
        num_users=num_users,
        num_movies=num_movies,
        mode="explicit",
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        model_save_path=ncf_save_path
    )
    
    # Plot comparisons
    plot_training_comparison(svd_history, ncf_history, mode="explicit", results_dir="results")

def run_implicit_pipeline(df, num_users, num_movies, args):
    """
    Executes SVD vs NCF training and evaluation for top-K recommendation (implicit feedback).
    """
    print("\n" + "="*70)
    print("RUNNING IMPLICIT FEEDBACK PIPELINE (TOP-10 RECOMMENDATION)")
    print("="*70)
    
    train_df, test_df = split_data_implicit(df)
    
    train_loader, test_instances = get_dataloaders(
        train_df, test_df, num_users, num_movies,
        mode="implicit", batch_size=args.batch_size, num_negatives=args.neg_ratio
    )
    
    # ---------------- 1. Train SVD Model ----------------
    print("\n--- Training SVD (Matrix Factorization) Model ---")
    svd_model = SVDModel(num_users=num_users, num_items=num_movies, embedding_dim=args.embedding_dim)
    svd_save_path = os.path.join("models", "best_svd_implicit.pth")
    svd_history = train_model(
        model=svd_model,
        train_loader=train_loader,
        val_data=test_instances,
        num_users=num_users,
        num_movies=num_movies,
        mode="implicit",
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        model_save_path=svd_save_path
    )
    
    # ---------------- 2. Train NCF Model ----------------
    print("\n--- Training NCF (NeuMF) Model ---")
    mlp_layers = [args.embedding_dim * 2, args.embedding_dim, args.embedding_dim // 2]
    ncf_model = NCFModel(
        num_users=num_users, 
        num_items=num_movies, 
        latent_dim_gmf=args.embedding_dim, 
        latent_dim_mlp=args.embedding_dim,
        mlp_layers=mlp_layers
    )
    ncf_save_path = os.path.join("models", "best_ncf_implicit.pth")
    ncf_history = train_model(
        model=ncf_model,
        train_loader=train_loader,
        val_data=test_instances,
        num_users=num_users,
        num_movies=num_movies,
        mode="implicit",
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        model_save_path=ncf_save_path
    )
    
    # Plot comparisons
    plot_training_comparison(svd_history, ncf_history, mode="implicit", results_dir="results")

def main():
    args = parse_args()
    
    # 1. Download and Extract MovieLens 1M dataset
    ml_1m_path = download_and_extract(data_dir="data")
    
    # 2. Load dataset
    df = load_ratings(ml_1m_path)
    
    # If quick-mode, downsample the dataset significantly for rapid validation
    if args.quick_mode:
        print("\n[Quick Mode] Enabled. Downsampling data to users <= 300 and running 1 epoch.")
        df = df[df["userId"] <= 300].copy()
        args.epochs = 1
        args.patience = 1
        ablation_epochs = 1
    else:
        ablation_epochs = 3 # Fast-running ablation epochs for full mode (early stopped anyway)

    # 3. Preprocess mappings
    df, num_users, num_movies = preprocess_data(df)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # 4. Execute Pipelines
    if args.mode in ["explicit", "both"]:
        run_explicit_pipeline(df, num_users, num_movies, args)
        
    if args.mode in ["implicit", "both"]:
        run_implicit_pipeline(df, num_users, num_movies, args)
        
        # Run ablation studies only for implicit ranking recommendation (since NCF is designed for this)
        if not args.skip_ablation:
            train_df, test_df = split_data_implicit(df)
            run_all_ablation_studies(
                train_df=train_df, 
                test_df=test_df, 
                num_users=num_users, 
                num_movies=num_movies, 
                device=args.device, 
                results_dir="results",
                epochs=ablation_epochs,
                quick_mode=args.quick_mode
            )
            plot_ablation_results(results_dir="results")
            
    print("\n" + "="*50)
    print("ALL RUNS COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
