"""
Book Embedding System — PyTorch LSTM

Trains an LSTM-based neural network to learn meaningful book embeddings from descriptions.
Similar books (based on ratings/genre) are pushed to have similar embeddings.

Usage:
    python train_embeddings.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG = {
    "data_path": "books.csv",
    "model_dir": "model",
    "vocab_size": 5000,
    "embed_dim": 100,           # Word embedding dimension
    "hidden_dim": 256,          # LSTM hidden state dimension
    "embedding_dim": 128,       # Output embedding dimension
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 64,           # Increased batch size
    "learning_rate": 0.0005,    # Slightly lower learning rate
    "num_epochs": 20,           # More epochs with early stopping
    "max_seq_len": 150,
    "margin": 0.5,              # Lower margin for better convergence
    "patience": 5,              # Early stopping patience
}


# ==============================================================================
# Vocabulary Builder
# ==============================================================================

class Vocabulary:
# ... (rest of Vocabulary class)
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
    
    def build(self, texts):
        """Build vocab from texts."""
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)
        
        # Keep top words
        for idx, (word, _) in enumerate(counter.most_common(self.max_size - 2), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"  Vocabulary: {len(self.word2idx)} words")
        return self
    
    def encode(self, text, max_len=None):
        """Convert text to token indices."""
        tokens = text.lower().split()
        indices = [self.word2idx.get(t, 1) for t in tokens]
        
        if max_len and len(indices) > max_len:
            indices = indices[:max_len]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def save(self, path):
        """Save vocabulary."""
        with open(path, "wb") as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
                "max_size": self.max_size,
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load vocabulary."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls(max_size=data["max_size"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        return vocab


# ==============================================================================
# Dataset
# ==============================================================================

class BookPairDataset(Dataset):
    """Creates (anchor, positive, negative) samples for triplet learning."""
    
    def __init__(self, descriptions, vocab, max_len=150):
        self.descriptions = descriptions
        self.vocab = vocab
        self.max_len = max_len
        
        # Build TF-IDF for content-based similarity
        print("  Computing TF-IDF for better positive pair generation...")
        tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        tfidf_matrix = tfidf.fit_transform(descriptions)
        self.sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Create triplet indices
        self._build_pairs()
    
    def _build_pairs(self):
        """Build anchor, positive, and negative triplets."""
        self.pairs = []
        num_books = len(self.descriptions)
        
        for i in range(num_books):
            # Positives: Top 10% most similar books by TF-IDF (excluding self)
            sim_scores = self.sim_matrix[i].copy()
            sim_scores[i] = -1  # Ignore self
            
            # Sort indices by similarity
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # Pick a positive from top 20 similar books
            positives = sorted_indices[:20]
            
            # Negatives: Pick from the bottom 50%
            negatives = sorted_indices[num_books // 2:]
            
            if len(positives) > 0 and len(negatives) > 0:
                # Add multiple triplets per book to increase dataset size
                for _ in range(3):
                    pos_idx = np.random.choice(positives)
                    neg_idx = np.random.choice(negatives)
                    self.pairs.append((i, pos_idx, neg_idx))
        
        print(f"  Created {len(self.pairs)} triplet pairs based on content similarity")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.pairs[idx]
        
        anchor = self.vocab.encode(self.descriptions[anchor_idx], self.max_len)
        positive = self.vocab.encode(self.descriptions[pos_idx], self.max_len)
        negative = self.vocab.encode(self.descriptions[neg_idx], self.max_len)
        
        return anchor, positive, negative


def collate_fn(batch):
    """Pad sequences in batch."""
    anchors, positives, negatives = zip(*batch)
    
    anchors_padded = pad_sequence(anchors, batch_first=True, padding_value=0)
    positives_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    negatives_padded = pad_sequence(negatives, batch_first=True, padding_value=0)
    
    return anchors_padded, positives_padded, negatives_padded


# ==============================================================================
# Model
# ==============================================================================

class BookEmbeddingModel(nn.Module):
    """LSTM-based book description encoder."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_dim, num_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Project LSTM output to embedding dimension
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) token indices
        
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Take last layer forward and backward hidden states
        last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Project to embedding space
        embeddings = self.fc(last_hidden)  # (batch, embedding_dim)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


# ==============================================================================
# Loss Function
# ==============================================================================

class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (batch_size, embedding_dim)
            positive: (batch_size, embedding_dim)
            negative: (batch_size, embedding_dim)
        """
        # Distance between anchor and positive (should be small)
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)
        
        # Distance between anchor and negative (should be large)
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)
        
        # Triplet loss
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        return loss.mean()


# ==============================================================================
# Training
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        # Compute loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
    
    return total_loss / len(loader)


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    print("BOOK EMBEDDING SYSTEM — LSTM + TRIPLET LOSS")
    print("=" * 70)
    
    # Create model directory
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/6] Loading books dataset...")
    
    try:
        books_df = pd.read_csv(CONFIG["data_path"])
        print(f"  ✓ Loaded {len(books_df)} books")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return
    
    # Clean descriptions
    books_df["Description"] = books_df["Description"].fillna("")
    books_df = books_df[books_df["Description"].str.len() > 10]
    
    descriptions = books_df["Description"].tolist()
    ratings = books_df["Rating"].tolist()
    
    print(f"  ✓ Using {len(descriptions)} books with descriptions")
    
    # =========================================================================
    # Step 2: Build vocabulary
    # =========================================================================
    print("\n[2/6] Building vocabulary...")
    
    vocab = Vocabulary(max_size=CONFIG["vocab_size"])
    vocab.build(descriptions)
    vocab.save(os.path.join(CONFIG["model_dir"], "vocab.pkl"))
    
    # =========================================================================
    # Step 3: Create dataset
    # =========================================================================
    print("\n[3/6] Creating training dataset...")
    
    dataset = BookPairDataset(descriptions, vocab, max_len=CONFIG["max_seq_len"])
    
    if len(dataset) == 0:
        print("  ✗ No valid pairs created. Check your data.")
        return
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"  ✓ Train: {len(train_dataset)} pairs | Val: {len(val_dataset)} pairs")
    print(f"  ✓ Train batches: {len(train_loader)}")
    
    # =========================================================================
    # Step 4: Build model
    # =========================================================================
    print("\n[4/6] Building model...")
    
    model = BookEmbeddingModel(
        vocab_size=len(vocab.word2idx),
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        embedding_dim=CONFIG["embedding_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model parameters: {total_params:,}")
    
    # =========================================================================
    # Step 5: Training
    # =========================================================================
    print("\n[5/6] Training...")
    print("-" * 70)
    
    criterion = TripletLoss(margin=CONFIG["margin"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    best_val_loss = float("inf")
    training_history = []
    
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"  Epoch {epoch}/{CONFIG['num_epochs']:2d} "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        training_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CONFIG,
                "vocab_size": len(vocab.word2idx),
            }, os.path.join(CONFIG["model_dir"], "embedding_model.pt"))
            print(f"    ✓ Best model saved")
    
    # =========================================================================
    # Step 6: Save artifacts
    # =========================================================================
    print("\n[6/6] Saving artifacts...")
    
    # Save training history
    with open(os.path.join(CONFIG["model_dir"], "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Compute embeddings for all books
    print("  Computing embeddings for all books...")
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for desc in descriptions:
            tokens = vocab.encode(desc, CONFIG["max_seq_len"]).unsqueeze(0).to(device)
            emb = model(tokens).cpu().numpy()[0]
            all_embeddings.append(emb)
    
    all_embeddings = np.array(all_embeddings)
    np.save(os.path.join(CONFIG["model_dir"], "book_embeddings.npy"), all_embeddings)
    
    # Save metadata
    metadata = {
        "titles": books_df["Title"].tolist(),
        "descriptions": books_df["Description"].tolist(),
        "prices": books_df.get("Price", [None] * len(books_df)).tolist(),
        "ratings": books_df.get("Rating", [None] * len(books_df)).tolist(),
        "cover_urls": books_df.get("Cover_URL", [None] * len(books_df)).tolist(),
    }
    with open(os.path.join(CONFIG["model_dir"], "books_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"  ✓ Saved {len(all_embeddings)} embeddings")
    print(f"  ✓ Saved metadata")
    
    # Save stats
    stats = {
        "num_books": len(descriptions),
        "vocab_size": len(vocab.word2idx),
        "embedding_dim": CONFIG["embedding_dim"],
        "best_val_loss": float(best_val_loss),
        "model_type": "LSTM Embedding",
    }
    with open(os.path.join(CONFIG["model_dir"], "model_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✓ EMBEDDING MODEL TRAINED!")
    print("=" * 70)
    print(f"Model: embedding_model.pt")
    print(f"Embeddings: book_embeddings.npy ({len(all_embeddings)} books)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nYou can now run: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
