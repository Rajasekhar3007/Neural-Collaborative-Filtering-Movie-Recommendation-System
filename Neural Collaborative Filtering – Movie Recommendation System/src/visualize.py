import os
import pandas as pd
import matplotlib.pyplot as plt

# Custom plotting theme for elegant, publication-quality graphics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

def plot_training_comparison(svd_history, ncf_history, mode="explicit", results_dir="results"):
    """
    Plots the training loss and validation metrics for SVD vs NCF models.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    if mode == "explicit":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # 1. Training Loss
        axes[0].plot(svd_history["train_loss"], label="SVD (Matrix Factorization)", color="#E15759", linewidth=2.5, marker='o')
        axes[0].plot(ncf_history["train_loss"], label="NCF (NeuMF)", color="#4E79A7", linewidth=2.5, marker='s')
        axes[0].set_title("Training Loss Comparison (MSE)", fontsize=13, fontweight='bold', pad=15)
        axes[0].set_xlabel("Epoch", fontsize=11, labelpad=8)
        axes[0].set_ylabel("Loss", fontsize=11, labelpad=8)
        axes[0].grid(True)
        axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
        
        # 2. RMSE
        axes[1].plot(svd_history["val_metric"], label="SVD (Matrix Factorization)", color="#E15759", linewidth=2.5, marker='o')
        axes[1].plot(ncf_history["val_metric"], label="NCF (NeuMF)", color="#4E79A7", linewidth=2.5, marker='s')
        axes[1].set_title("Validation RMSE Comparison (Lower is Better)", fontsize=13, fontweight='bold', pad=15)
        axes[1].set_xlabel("Epoch", fontsize=11, labelpad=8)
        axes[1].set_ylabel("RMSE", fontsize=11, labelpad=8)
        axes[1].grid(True)
        axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
        
        plt.suptitle("Explicit Feedback: Rating Prediction (SVD vs. NCF)", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(results_dir, "svd_vs_ncf_explicit.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Visualize] Rating prediction plots saved to {save_path}")
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # 1. Training Loss (BCE)
        axes[0].plot(svd_history["train_loss"], label="SVD (Matrix Factorization)", color="#E15759", linewidth=2.5, marker='o')
        axes[0].plot(ncf_history["train_loss"], label="NCF (NeuMF)", color="#4E79A7", linewidth=2.5, marker='s')
        axes[0].set_title("Training Loss (Binary Cross Entropy)", fontsize=12, fontweight='bold', pad=12)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].grid(True)
        axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
        
        # 2. Hit Rate @ 10
        axes[1].plot(svd_history["val_metric"], label="SVD", color="#E15759", linewidth=2.5, marker='o')
        axes[1].plot(ncf_history["val_metric"], label="NCF", color="#4E79A7", linewidth=2.5, marker='s')
        axes[1].set_title("Validation Hit Rate @ 10 (Higher is Better)", fontsize=12, fontweight='bold', pad=12)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("HR@10", fontsize=11)
        axes[1].grid(True)
        axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
        
        # 3. NDCG @ 10
        axes[2].plot(svd_history["val_ndcg"], label="SVD", color="#E15759", linewidth=2.5, marker='o')
        axes[2].plot(ncf_history["val_ndcg"], label="NCF", color="#4E79A7", linewidth=2.5, marker='s')
        axes[2].set_title("Validation NDCG @ 10 (Higher is Better)", fontsize=12, fontweight='bold', pad=12)
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("NDCG@10", fontsize=11)
        axes[2].grid(True)
        axes[2].legend(frameon=True, facecolor='white', edgecolor='none')
        
        plt.suptitle("Implicit Feedback: Top-10 Recommendation (SVD vs. NCF)", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(results_dir, "svd_vs_ncf_implicit.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Visualize] Top-K recommendation plots saved to {save_path}")

def plot_ablation_results(results_dir="results"):
    """
    Reads the ablation results from CSVs and saves a 1x3 grid visualization.
    """
    embedding_csv = os.path.join(results_dir, "ablation_embedding.csv")
    depth_csv = os.path.join(results_dir, "ablation_depth.csv")
    neg_ratio_csv = os.path.join(results_dir, "ablation_neg_ratio.csv")
    
    if not (os.path.exists(embedding_csv) and os.path.exists(depth_csv) and os.path.exists(neg_ratio_csv)):
        print("[Visualize] Ablation CSV files not found. Skipping ablation plot generation.")
        return
        
    df_embed = pd.read_csv(embedding_csv)
    df_depth = pd.read_csv(depth_csv)
    df_neg = pd.read_csv(neg_ratio_csv)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. Embedding Size Ablation
    axes[0].plot(df_embed["embedding_size"], df_embed["hr_at_10"], color="#2F4F4F", marker='o', linewidth=2.5, label="HR@10")
    axes[0].plot(df_embed["embedding_size"], df_embed["ndcg_at_10"], color="#708090", marker='s', linewidth=2, linestyle='--', label="NDCG@10")
    axes[0].set_title("Ablation: Embedding Size", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel("Latent Dimension (GMF/MLP Size)", fontsize=11)
    axes[0].set_ylabel("Metric Score", fontsize=11)
    axes[0].set_xticks(df_embed["embedding_size"])
    axes[0].grid(True)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
    
    # 2. MLP Depth Ablation
    axes[1].plot(df_depth["num_layers"], df_depth["hr_at_10"], color="#D95F02", marker='o', linewidth=2.5, label="HR@10")
    axes[1].plot(df_depth["num_layers"], df_depth["ndcg_at_10"], color="#E6AB02", marker='s', linewidth=2, linestyle='--', label="NDCG@10")
    axes[1].set_title("Ablation: MLP Layer Depth", fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel("Number of MLP Hidden Layers", fontsize=11)
    axes[1].set_ylabel("Metric Score", fontsize=11)
    axes[1].set_xticks(df_depth["num_layers"])
    axes[1].grid(True)
    axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
    
    # 3. Negative Sampling Ratio Ablation
    axes[2].plot(df_neg["neg_ratio"], df_neg["hr_at_10"], color="#7570B3", marker='o', linewidth=2.5, label="HR@10")
    axes[2].plot(df_neg["neg_ratio"], df_neg["ndcg_at_10"], color="#E7298A", marker='s', linewidth=2, linestyle='--', label="NDCG@10")
    axes[2].set_title("Ablation: Negative Sampling Ratio", fontsize=12, fontweight='bold', pad=12)
    axes[2].set_xlabel("Negative Samples per Positive Interaction", fontsize=11)
    axes[2].set_ylabel("Metric Score", fontsize=11)
    axes[2].set_xticks(df_neg["neg_ratio"])
    axes[2].grid(True)
    axes[2].legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.suptitle("NCF/NeuMF Parameter Ablation Studies (HR@10 & NDCG@10)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "ncf_ablation_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualize] Ablation results plot saved to {save_path}")
