import torch
import torch.nn as nn

class SVDModel(nn.Module):
    """
    SVD (Matrix Factorization) Model in PyTorch.
    Predicts rating or interaction probability using user and item embeddings + biases.
    """
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(SVDModel, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Learnable global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings with small normal distribution
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embeddings(user_indices)
        item_embed = self.item_embeddings(item_indices)
        
        # Dot product between user and item embeddings
        dot = torch.sum(user_embed * item_embed, dim=1, keepdim=True)
        
        # Add biases
        u_bias = self.user_bias(user_indices)
        i_bias = self.item_bias(item_indices)
        
        prediction = dot + u_bias + i_bias + self.global_bias
        return prediction.squeeze()


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering (NCF / NeuMF) Model.
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
    """
    def __init__(self, num_users, num_items, latent_dim_gmf=32, latent_dim_mlp=32, 
                 mlp_layers=[64, 32, 16], dropout=0.2):
        """
        Args:
            num_users: Number of users in the dataset.
            num_items: Number of items in the dataset.
            latent_dim_gmf: Latent dimension for the GMF branch.
            latent_dim_mlp: Latent dimension for the MLP branch (usually equals GMF dim).
            mlp_layers: Dimensions of MLP hidden layers (first layer must be latent_dim_mlp * 2).
            dropout: Dropout rate for MLP layers.
        """
        super(NCFModel, self).__init__()
        
        # ---------------- GMF Branch ----------------
        self.user_embeddings_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embeddings_gmf = nn.Embedding(num_items, latent_dim_gmf)
        
        # ---------------- MLP Branch ----------------
        self.user_embeddings_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embeddings_mlp = nn.Embedding(num_items, latent_dim_mlp)
        
        # Check if first layer matches concatenated user & item embeddings
        assert mlp_layers[0] == latent_dim_mlp * 2, "First layer size of mlp_layers must be latent_dim_mlp * 2"
        
        # MLP network layers
        mlp_modules = []
        for i in range(len(mlp_layers) - 1):
            mlp_modules.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            mlp_modules.append(nn.BatchNorm1d(mlp_layers[i+1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
        
        self.mlp_network = nn.Sequential(*mlp_modules)
        
        # ---------------- Prediction Layer ----------------
        # Concatenate GMF output (dimension: latent_dim_gmf) and MLP output (dimension: mlp_layers[-1])
        prediction_dim = latent_dim_gmf + mlp_layers[-1]
        self.prediction_layer = nn.Linear(prediction_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        # Xavier/Normal Initialization for NCF weights
        nn.init.normal_(self.user_embeddings_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embeddings_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embeddings_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embeddings_mlp.weight, std=0.01)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_indices, item_indices):
        # --- GMF ---
        gmf_user = self.user_embeddings_gmf(user_indices)
        gmf_item = self.item_embeddings_gmf(item_indices)
        gmf_out = gmf_user * gmf_item
        
        # --- MLP ---
        mlp_user = self.user_embeddings_mlp(user_indices)
        mlp_item = self.item_embeddings_mlp(item_indices)
        mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp_network(mlp_in)
        
        # --- Fusion ---
        fusion = torch.cat([gmf_out, mlp_out], dim=-1)
        prediction = self.prediction_layer(fusion)
        
        return prediction.squeeze()
