import os
import zipfile
import requests
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_and_extract(data_dir="data"):
    """
    Downloads and extracts the MovieLens 1M dataset if not already present.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extract_path = os.path.join(data_dir, "ml-1m")
    
    # If already extracted, skip
    if os.path.exists(os.path.join(extract_path, "ratings.dat")):
        print("[Data] MovieLens 1M dataset already exists and is extracted.")
        return extract_path

    # Download zip file
    if not os.path.exists(zip_path):
        print(f"[Data] Downloading MovieLens 1M from {MOVIELENS_URL}...")
        response = requests.get(MOVIELENS_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("[Data] Download complete.")

    # Extract zip file
    print("[Data] Extracting ml-1m.zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    print("[Data] Extraction complete.")
    
    # Clean up zip file to save space
    try:
        os.remove(zip_path)
    except OSError:
        pass
        
    return extract_path

def load_ratings(data_path):
    """
    Loads ratings.dat from the MovieLens 1M dataset.
    """
    ratings_file = os.path.join(data_path, "ratings.dat")
    print(f"[Data] Loading ratings from {ratings_file}...")
    
    # MovieLens 1M ratings.dat: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(
        ratings_file,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1"
    )
    return df

def preprocess_data(df):
    """
    Encodes userId and movieId into contiguous index integers [0, N-1].
    """
    print("[Data] Preprocessing user and movie IDs...")
    
    # Encode Users
    user_unique = df["userId"].unique()
    user_to_idx = {user: idx for idx, user in enumerate(user_unique)}
    df["user_idx"] = df["userId"].map(user_to_idx)
    
    # Encode Movies
    movie_unique = df["movieId"].unique()
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_unique)}
    df["movie_idx"] = df["movieId"].map(movie_to_idx)
    
    num_users = len(user_unique)
    num_movies = len(movie_unique)
    
    print(f"[Data] Dataset statistics: {num_users} users, {num_movies} movies, {len(df)} ratings.")
    return df, num_users, num_movies

def split_data_explicit(df, test_ratio=0.2):
    """
    Splits the data for rating prediction (explicit feedback) randomly.
    """
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    test_size = int(len(df) * test_ratio)
    
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    return train_df, test_df

def split_data_implicit(df):
    """
    Leave-one-out split for top-K recommendation (implicit feedback).
    The latest rating (by timestamp) for each user is used as the test case,
    and all other ratings are used for training.
    """
    print("[Data] Performing Leave-One-Out split for ranking evaluation...")
    df = df.sort_values(by=["user_idx", "timestamp"])
    
    # Group by user_idx and get the index of the last element
    test_df = df.groupby("user_idx").tail(1).copy()
    
    # Train set is all elements not in the test set
    train_df = df.drop(test_df.index).copy()
    
    return train_df, test_df

class RatingDataset(Dataset):
    """
    PyTorch Dataset for explicit feedback (rating prediction).
    """
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class ImplicitTrainDataset(Dataset):
    """
    PyTorch Dataset for implicit feedback training.
    Performs negative sampling dynamically at the start of each epoch.
    """
    def __init__(self, train_df, num_users, num_movies, num_negatives=4):
        self.train_df = train_df
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_negatives = num_negatives
        
        # Store user interactions as a set for O(1) lookups
        self.user_pos_items = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        
        self.users = []
        self.items = []
        self.labels = []
        
        # Initial sampling
        self.sample_negatives()

    def sample_negatives(self):
        """
        Samples N negative items for each positive rating.
        """
        users_list = []
        items_list = []
        labels_list = []
        
        # Iterate over all training interactions
        user_idxs = self.train_df["user_idx"].values
        movie_idxs = self.train_df["movie_idx"].values
        
        for u, i in zip(user_idxs, movie_idxs):
            # Positive sample
            users_list.append(u)
            items_list.append(i)
            labels_list.append(1.0)
            
            # Negative samples
            pos_items = self.user_pos_items[u]
            for _ in range(self.num_negatives):
                neg_item = np.random.randint(0, self.num_movies)
                while neg_item in pos_items:
                    neg_item = np.random.randint(0, self.num_movies)
                users_list.append(u)
                items_list.append(neg_item)
                labels_list.append(0.0)
                
        self.users = torch.tensor(users_list, dtype=torch.long)
        self.items = torch.tensor(items_list, dtype=torch.long)
        self.labels = torch.tensor(labels_list, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def get_implicit_test_instances(test_df, train_df, num_movies, num_negatives=99):
    """
    Generates test instances for ranking evaluation.
    For each user, returns (positive_item, [99 negative_items]).
    """
    print("[Data] Generating test instances with negative sampling (99 per user)...")
    user_pos_items = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    # Add test items to user_pos_items to ensure we don't sample them
    test_user_pos = test_df.set_index("user_idx")["movie_idx"].to_dict()
    
    test_data = {}
    
    for _, row in test_df.iterrows():
        u = row["user_idx"]
        pos_item = row["movie_idx"]
        
        # Collect all items user interacted with
        interacted_items = user_pos_items.get(u, set()) | {pos_item}
        
        neg_items = []
        for _ in range(num_negatives):
            neg_item = np.random.randint(0, num_movies)
            while neg_item in interacted_items or neg_item in neg_items:
                neg_item = np.random.randint(0, num_movies)
            neg_items.append(neg_item)
            
        test_data[u] = {
            "pos": pos_item,
            "negs": neg_items
        }
        
    return test_data

def get_dataloaders(train_df, test_df, num_users, num_movies, mode="explicit", batch_size=256, num_negatives=4):
    """
    Returns dataloaders for the training and testing sets.
    """
    if mode == "explicit":
        train_dataset = RatingDataset(train_df)
        test_dataset = RatingDataset(test_df)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
        
    elif mode == "implicit":
        train_dataset = ImplicitTrainDataset(
            train_df, num_users, num_movies, num_negatives=num_negatives
        )
        test_instances = get_implicit_test_instances(test_df, train_df, num_movies, num_negatives=99)
        
        # We don't use a standard DataLoader for test evaluation since leave-one-out
        # evaluation calculates ranking per user. We will evaluate custom test instances directly.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_instances
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
