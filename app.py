"""
Book Recommendation System Dashboard — Streamlit App (with LSTM Embeddings)

Interactive demo for the LSTM + Triplet Loss book recommender.
Features:
  1. Search for a book by title
  2. Get personalized recommendations based on semantic similarity
  3. Browse books and explore similarity scores
  4. View recommender statistics

Usage:
    streamlit run app_embeddings.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
import requests

# ==============================================================================
# Page Config
# ==============================================================================

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS
# ==============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }

    .book-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .book-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }

    .book-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .book-meta {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }

    .book-rating {
        font-size: 0.95rem;
        color: #f59e0b;
        font-weight: 500;
    }

    .similarity-score {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .metric-card h3 {
        color: #475569;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .metric-card p {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Model Loading
# ==============================================================================

MODEL_DIR = "model"
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "embedding_model.pt")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "book_embeddings.npy")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "books_metadata.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "model_stats.json")


class Vocabulary:
    """Simple vocabulary class for tokenization."""
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_count = {}

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_length=150):
        """Convert text to token indices."""
        words = text.lower().split()
        tokens = []
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx["<UNK>"])
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)


class BookEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=256, embedding_dim=128, num_layers=2, dropout=0.3):
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
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        """x: (batch_size, seq_length)"""
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)  # lstm_out: (batch, seq, hidden*2)
        
        # Take last layer forward and backward hidden states
        last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        embedding = self.fc(last_hidden)  # (batch_size, embedding_dim)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


@st.cache_resource
def load_recommender():
    """Load embedding model, embeddings, vocab, and book metadata."""
    if not all(os.path.exists(p) for p in [EMBEDDING_MODEL_PATH, EMBEDDINGS_PATH, VOCAB_PATH, METADATA_PATH]):
        return None, None, None, None, None

    # Load vocab
    with open(VOCAB_PATH, "rb") as f:
        raw_vocab = pickle.load(f)

    # Normalize vocab: support dict or saved Vocabulary object
    if isinstance(raw_vocab, dict):
        word2idx = raw_vocab.get('word2idx', raw_vocab)

        class VocabWrapper:
            def __init__(self, word2idx):
                self.word2idx = word2idx

            def encode(self, text, max_length=150):
                words = text.lower().split()
                tokens = [self.word2idx.get(w, self.word2idx.get('<UNK>', 1)) for w in words]
                if len(tokens) < max_length:
                    tokens += [0] * (max_length - len(tokens))
                else:
                    tokens = tokens[:max_length]
                return torch.tensor(tokens, dtype=torch.long)

        vocab = VocabWrapper(word2idx)
    else:
        vocab = raw_vocab

    # Load embeddings
    book_embeddings = np.load(EMBEDDINGS_PATH)  # (1000, 128)

    # Load metadata
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Load model
    device = torch.device("cpu")
    vocab_size = len(vocab.word2idx) if hasattr(vocab, 'word2idx') else getattr(vocab, 'vocab_size', None)
    if vocab_size is None:
        vocab_size = CONFIG.get('vocab_size', 5000) if 'CONFIG' in globals() else 5000

    model = BookEmbeddingModel(vocab_size=vocab_size)
    state = torch.load(EMBEDDING_MODEL_PATH, map_location=device)
    # support checkpoints with 'model_state_dict' or 'state_dict'
    if isinstance(state, dict):
        sd = state.get('model_state_dict') or state.get('state_dict') or state
        try:
            model.load_state_dict(sd)
        except Exception:
            # if loading fails, attempt to find inner dict
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                model.load_state_dict(sd['model_state_dict'])
            else:
                raise
    else:
        # saved entire model
        model = state

    model.to(device).eval()

    stats = {}
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r") as f:
            stats = json.load(f)

    return model, vocab, book_embeddings, metadata, stats


def get_recommendations(book_title, metadata, book_embeddings, top_n=8):
    """Find top N most similar books to the given book using embedding similarity."""
    try:
        # Find the book index
        titles = metadata["titles"]
        book_idx = titles.index(book_title)
    except ValueError:
        return None

    # Get embedding for this book
    book_embedding = book_embeddings[book_idx]  # (128,)
    
    # Compute cosine similarity with all other books
    # Normalize embeddings are already L2 normalized, so just use dot product
    similarities = np.dot(book_embeddings, book_embedding)  # (1000,)
    
    # Get top N similar books (excluding the book itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    similar_scores = similarities[similar_indices]

    recommendations = []
    for idx, score in zip(similar_indices, similar_scores):
        recommendations.append({
            "title": metadata["titles"][idx],
            "description": metadata["descriptions"][idx],
            "price": metadata["prices"][idx] if idx < len(metadata["prices"]) else None,
            "rating": metadata["ratings"][idx] if idx < len(metadata["ratings"]) else None,
            "cover_url": metadata["cover_urls"][idx] if "cover_urls" in metadata and idx < len(metadata["cover_urls"]) else None,
            "similarity": float(score),
        })

    return recommendations


def get_custom_recommendations(custom_text, model, vocab, book_embeddings, metadata, top_n=8):
    """Get recommendations based on custom text input."""
    if not custom_text or not custom_text.strip():
        return []
    
    try:
        # Encode custom text
        custom_tokens = vocab.encode(custom_text, max_length=150)
        # Ensure tensor type and device
        if not isinstance(custom_tokens, torch.LongTensor):
            custom_tokens = custom_tokens.long()
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')
        custom_tokens = custom_tokens.to(device)
        
        # Debug: capture shapes before forward
        try:
            seq_shape = tuple(custom_tokens.unsqueeze(0).shape)
            lstm_info = None
            if hasattr(model, 'lstm'):
                lstm = model.lstm
                # infer hidden_size and bidirectional
                hidden_size = getattr(lstm, 'hidden_size', None)
                bidir = getattr(lstm, 'bidirectional', False)
                num_directions = 2 if bidir else 1
                lstm_info = (hidden_size, num_directions)
            # run forward
            with torch.no_grad():
                custom_embedding = model(custom_tokens.unsqueeze(0))  # (1, embedding_dim)
                custom_embedding = custom_embedding.squeeze(0).cpu().numpy()
        except Exception as inner_e:
            # Build detailed debug message
            shapes = {
                'custom_tokens_shape': tuple(custom_tokens.shape),
                'seq_shape': seq_shape if 'seq_shape' in locals() else None,
                'lstm_info': lstm_info,
            }
            # Also include first linear weight shapes if present
            fc_shapes = {}
            if hasattr(model, 'fc'):
                try:
                    for i, layer in enumerate(model.fc):
                        if hasattr(layer, 'weight'):
                            fc_shapes[f'fc[{i}].weight'] = tuple(layer.weight.shape)
                except Exception:
                    pass
            debug_msg = f"Forward error: {inner_e}; shapes={shapes}; fc_shapes={fc_shapes}"
            st.error(f"Error processing text: {str(inner_e)}")
            print(debug_msg)
            return []
        
        # Compute cosine similarity with all books
        similarities = np.dot(book_embeddings, custom_embedding)
        
        # Get top N similar books
        similar_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                "title": metadata["titles"][idx],
                "description": metadata["descriptions"][idx],
                "price": metadata["prices"][idx] if idx < len(metadata["prices"]) else None,
                "rating": metadata["ratings"][idx] if idx < len(metadata["ratings"]) else None,
                "cover_url": metadata["cover_urls"][idx] if "cover_urls" in metadata and idx < len(metadata["cover_urls"]) else None,
                "similarity": float(similarities[idx]),
            })
        
        return recommendations
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return []


@st.cache_data
def load_image_from_url(url):
    """Load and cache book cover image from URL."""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception:
        pass
    return None


# ==============================================================================
# App Layout
# ==============================================================================

# Header
st.markdown('<h1 class="main-title">📚 Book Recommendation System</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find similar books based on semantic similarity — powered by LSTM + Triplet Loss Embeddings</p>',
            unsafe_allow_html=True)

# Load recommender
model, vocab, book_embeddings, metadata, stats = load_recommender()
recommender_loaded = model is not None

# Sidebar
st.sidebar.title("🔍 Navigation")
st.sidebar.markdown("---")

if recommender_loaded:
    num_books = stats.get("num_books", 0)
    embedding_dim = stats.get("embedding_dim", 128)
    st.sidebar.success(f"✅ Recommender loaded\n\n📚 Books: {num_books}\n🧠 Embeddings: {embedding_dim}D")
else:
    st.sidebar.error("❌ Recommender not found\n\nRun `python train_embeddings.py` first")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Project Files")
st.sidebar.markdown("""
- `train_embeddings.py` — Trains LSTM embeddings
- `scraper.py` — Data collection
- `app_embeddings.py` — This dashboard
- `books.csv` — Book dataset
""")

# Tabs
tab1, tab2 = st.tabs([
    "🎯 Get Recommendations",
    "📊 Statistics",
])


# ==============================================================================
# Tab 1: Get Recommendations
# ==============================================================================

with tab1:
    st.header("Find Similar Books")
    st.markdown("Enter custom text and get personalized recommendations based on semantic similarity.")

    if not recommender_loaded:
        st.error("⚠️ Recommender not loaded. Run `python train_embeddings.py` first.")
    else:
        # Number of recommendations slider (common for both methods)
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=3,
            max_value=12,
            value=8,
        )
        
        st.markdown("---")

        st.markdown("""
        **Enter a description of a fictional book or any text you'd like to find similar books for:**

        Example: "A dystopian novel about a young girl discovering magical powers in a technologically advanced society"
        """)

        custom_text = st.text_area(
            "📝 Enter book description or custom text:",
            placeholder="Type or paste a book description, synopsis, or any text here...",
            height=150,
            help="Describe a fictional book or provide any text to find similar books from our database"
        )

        if custom_text.strip():
            st.markdown("---")
            st.markdown("### 📚 Recommendations Based on Your Text")

            # Get recommendations based on custom text
            recommendations = get_custom_recommendations(
                custom_text,
                model,
                vocab,
                book_embeddings,
                metadata,
                top_n=num_recommendations
            )

            if recommendations:
                st.success(f"✅ Found {len(recommendations)} similar books!")

                for i, rec in enumerate(recommendations, 1):
                    col1, col2, col3 = st.columns([1, 4, 1])

                    # Cover image
                    with col1:
                        if rec['cover_url']:
                            img = load_image_from_url(rec['cover_url'])
                            if img:
                                st.image(img, width=100)
                            else:
                                st.markdown("<div style='width:100px; height:130px; background: #ddd; display:flex; align-items:center; justify-content:center; border-radius:8px; font-size:40px;'>📕</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='width:100px; height:130px; background: #ddd; display:flex; align-items:center; justify-content:center; border-radius:8px; font-size:40px;'>📕</div>", unsafe_allow_html=True)

                    # Book details
                    with col2:
                        st.markdown(f"""
                        <div class="book-card">
                            <div class="book-title">#{i} {rec['title']}</div>
                            <div class="book-meta">{rec['description'][:250]}...</div>
                            <div class="book-meta">
                                {"⭐ " + str(rec['rating']) if rec['rating'] else ""}
                                {"• " if rec['rating'] and rec['price'] else ""}
                                {"💷 " + str(rec['price']) if rec['price'] else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Similarity percentage
                    with col3:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div style="font-size: 24px; font-weight: bold; color: #2E86AB;">
                                {rec['similarity']:.0%}
                            </div>
                            <div style="font-size: 12px; color: #666;">Match</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")
            else:
                st.warning("No similar books found. Try using different keywords!")
        else:
            st.info("👆 Enter some text above to find similar books in our database")


# ==============================================================================
# Tab 2: Statistics
# ==============================================================================

with tab2:
    st.header("📊 Recommender Statistics")

    if not recommender_loaded:
        st.error("⚠️ Recommender not loaded. Run `python train_embeddings.py` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        st.markdown("### 🧠 Model Architecture & Parameters")
        
        # Calculate parameters for display
        def count_parameters(model):
            # Embedding params
            emb = model.embedding.num_embeddings * model.embedding.embedding_dim
            # LSTM params
            lstm = sum(p.numel() for p in model.lstm.parameters())
            # FC params
            fc = sum(p.numel() for p in model.fc.parameters())
            return emb, lstm, fc

        emb_p, lstm_p, fc_p = count_parameters(model)
        total_p = emb_p + lstm_p + fc_p
        
        col_arch1, col_arch2 = st.columns([3, 2])
        
        with col_arch1:
            st.markdown("#### Neural Flow & Activations")
            
            def layer_box(num, name, shape, details, act=None):
                act_html = f'<div style="color: #667eea; font-weight: 700; font-size: 0.75rem; margin-top: 4px;">⚡ Activation: {act}</div>' if act else ""
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-left: 5px solid #667eea; padding: 12px; border-radius: 8px; margin-bottom: 0px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <span style="font-weight: 700; color: #1e293b; font-size: 0.9rem;">{num}. {name}</span>
                        <span style="background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: monospace;">{shape}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">{details}</div>
                    {act_html}
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<div style="text-align: center; color: #cbd5e1; margin: 2px 0;">↓</div>', unsafe_allow_html=True)

            layer_box("01", "Input Layer", "1 x 150", "Tokenized text sequences (integer indices)")
            layer_box("02", "Embedding", f"{len(vocab.word2idx)} x 100", "Semantic word projection", "Linear / Lookup")
            layer_box("03", "Bi-LSTM", "150 x 512", "Temporal feature extraction (2-layer, bidirectional)", "Tanh / Sigmoid")
            layer_box("04", "Hidden Dense", "512 x 256", "High-level feature mapping", "ReLU")
            layer_box("05", "Dropout", "256", "Regularization (rate: 0.3)", "N/A")
            layer_box("06", "Output Projection", "256 x 128", "L2-normalized semantic embedding", "L2-Norm")

        with col_arch2:
            st.markdown("#### Parameter Distribution")
            
            # Pie chart for parameters
            fig_p = px.pie(
                values=[emb_p, lstm_p, fc_p],
                names=['Embedding', 'LSTM (Bi-Dir)', 'Dense Head'],
                hole=0.5,
                color_discrete_sequence=['#667eea', '#764ba2', '#a0aec0']
            )
            fig_p.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=250,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_p, use_container_width=True)
            
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0;">
                <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 5px;">Total Trainable Parameters</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1e293b;">{total_p:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Logic")
            st.caption("The model utilizes **Triplet Loss** during training to ensure that the Euclidean distance between content-similar books is minimized in the 128D embedding space.")

        st.markdown("---")
        st.markdown("""
        This recommendation system uses **LSTM-based learned embeddings** with **triplet loss** 
        to find semantically similar books.

        **Architecture:**
        1. **Input Encoding:** Book descriptions are tokenized and converted to token sequences
        2. **Embedding Layer:** Words are mapped to 100-dimensional vectors
        3. **Bi-LSTM Encoder:** Two-layer bidirectional LSTM with 256 hidden units
           - Captures sequential and contextual information from description
           - Forward and backward passes merged to understand text from both directions
        4. **Dense Projection:** Output projected to 128-dimensional space
           - Reduces dimensionality while preserving semantic information
        5. **Normalization:** L2 normalization ensures comparable similarity scores

        **Training:**
        - **Loss Function:** Triplet Loss for metric learning
           - Anchor: A book description
           - Positive: Description of a book with high TF-IDF similarity (content-based)
           - Negative: Description of a book with low TF-IDF similarity
        - **Objective:** Minimize distance between content-similar books, maximize between dissimilar
        - **Dataset:** ~3,000 triplet pairs generated from 1,000 books
        - **Epochs:** 20 with early stopping

        **Advantages:**
        - ✅ Learns semantic relationships: Captures meaning, not just keywords
        - ✅ Handles new text: Can encode custom descriptions not in training set
        - ✅ Deep learning: Uses neural networks as required by course
        - ✅ Scalable: Fast inference with O(1) similarity lookup
        - ✅ Interpretable: Embeddings capture meaningful book characteristics

        **Limitations:**
        - Recommendations based on descriptions, not reader sentiment
        - Limited to training set ratings for triplet generation
        - Doesn't improve with user feedback (no online learning)
        """)

        st.markdown("---")
        st.markdown("### Training History")
        
        if os.path.exists(os.path.join(MODEL_DIR, "training_history.json")):
            with open(os.path.join(MODEL_DIR, "training_history.json"), "r") as f:
                raw_history = json.load(f)

            # Normalize into a list of epoch dicts
            history = []
            if isinstance(raw_history, list):
                history = raw_history
            elif isinstance(raw_history, dict):
                # dict of lists -> convert to list of dicts
                # find length from any list value
                lengths = [len(v) for v in raw_history.values() if isinstance(v, list)]
                n = max(lengths) if lengths else 0
                for i in range(n):
                    entry = {}
                    for k, v in raw_history.items():
                        if isinstance(v, list) and i < len(v):
                            entry[k] = v[i]
                    history.append(entry)

            # helper to extract series by trying multiple key names
            def extract_series(history, keys):
                for k in keys:
                    vals = [h.get(k) for h in history if h.get(k) is not None]
                    if vals:
                        return vals
                return []

            train_loss = extract_series(history, ['train_loss', 'train_losses', 'loss'])
            val_loss = extract_series(history, ['val_loss', 'val_losses', 'val'])
            train_acc = extract_series(history, ['train_acc', 'train_accuracy', 'accuracy', 'acc'])
            val_acc = extract_series(history, ['val_acc', 'val_accuracy', 'val_accuracy', 'val_acc'])
            precision = extract_series(history, ['precision', 'val_precision', 'precision_score'])
            recall = extract_series(history, ['recall', 'val_recall', 'recall_score'])
            f1 = extract_series(history, ['f1', 'f1_score', 'val_f1'])

            epochs = list(range(1, max(len(train_loss), len(val_loss), len(train_acc), len(val_acc)) + 1))

            # Build subplot: losses on top, metrics underneath (if present)
            row_heights = [0.6, 0.4]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                row_heights=row_heights, subplot_titles=("Loss (Train vs Validation)", "Metrics"))

            if train_loss:
                fig.add_trace(go.Scatter(x=list(range(1, len(train_loss) + 1)), y=train_loss,
                                         mode='lines+markers', name='Train Loss', line=dict(color='#667eea', width=2),
                                         marker=dict(size=6)), row=1, col=1)
            if val_loss:
                fig.add_trace(go.Scatter(x=list(range(1, len(val_loss) + 1)), y=val_loss,
                                         mode='lines+markers', name='Validation Loss', line=dict(color='#764ba2', width=2),
                                         marker=dict(size=6)), row=1, col=1)

            # Add metrics traces (accuracy / precision / recall / f1) to row 2
            metric_present = False
            if train_acc or val_acc:
                metric_present = True
                if train_acc:
                    fig.add_trace(go.Scatter(x=list(range(1, len(train_acc) + 1)), y=train_acc,
                                             mode='lines+markers', name='Train Accuracy', line=dict(color='#06b6d4', width=2)), row=2, col=1)
                if val_acc:
                    fig.add_trace(go.Scatter(x=list(range(1, len(val_acc) + 1)), y=val_acc,
                                             mode='lines+markers', name='Val Accuracy', line=dict(color='#0ea5a4', width=2)), row=2, col=1)

            if precision:
                metric_present = True
                fig.add_trace(go.Scatter(x=list(range(1, len(precision) + 1)), y=precision,
                                         mode='lines+markers', name='Precision', line=dict(color='#f59e0b', width=2)), row=2, col=1)
            if recall:
                metric_present = True
                fig.add_trace(go.Scatter(x=list(range(1, len(recall) + 1)), y=recall,
                                         mode='lines+markers', name='Recall', line=dict(color='#ef4444', width=2)), row=2, col=1)
            if f1:
                metric_present = True
                fig.add_trace(go.Scatter(x=list(range(1, len(f1) + 1)), y=f1,
                                         mode='lines+markers', name='F1-score', line=dict(color='#7c3aed', width=2)), row=2, col=1)

            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Triplet Loss", row=1, col=1)
            if metric_present:
                fig.update_yaxes(title_text="Metric Value", row=2, col=1)
            else:
                # remove subtitle if no metrics
                fig.layout.annotations[1].text = "(No additional metrics found)"

            fig.update_layout(height=600, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Diagnostics: simple heuristics for overfitting / underfitting / stability
            diag_msgs = []
            if train_loss and val_loss:
                final_train = train_loss[-1]
                final_val = val_loss[-1]
                # Overfitting: val loss significantly higher than train loss and trend diverging
                if final_val - final_train > 0.02 and (np.mean(val_loss[-3:]) > np.mean(train_loss[-3:])):
                    diag_msgs.append(("Overfitting", "Validation loss is higher than training loss — possible overfitting."))
                # Underfitting: both losses high and not decreasing much
                if final_train > 0.5 and final_val > 0.5 and (np.mean(train_loss[:3]) - final_train) < 0.01:
                    diag_msgs.append(("Underfitting", "Both train and validation loss remain high — model may be underfitting."))
                # Stability: oscillation in validation loss
                if len(val_loss) >= 6:
                    recent_std = float(np.std(val_loss[-6:]))
                    recent_mean = float(np.mean(val_loss[-6:]))
                    if recent_std / (recent_mean + 1e-9) > 0.05:
                        diag_msgs.append(("Unstable", "Validation loss shows oscillation — training stability may be an issue."))

            # Display diagnostics
            if diag_msgs:
                for title, msg in diag_msgs:
                    st.warning(f"**{title}:** {msg}")
            else:
                st.success("No obvious overfitting/underfitting signals detected from available training history.")

            # Show final metric values if available
            metric_line = []
            if train_acc or val_acc:
                metric_line.append(f"Final Train Acc: {train_acc[-1]:.3f}" if train_acc else None)
                metric_line.append(f"Final Val Acc: {val_acc[-1]:.3f}" if val_acc else None)
            if precision:
                metric_line.append(f"Precision: {precision[-1]:.3f}")
            if recall:
                metric_line.append(f"Recall: {recall[-1]:.3f}")
            if f1:
                metric_line.append(f"F1: {f1[-1]:.3f}")

            if any(metric_line):
                metric_line = [m for m in metric_line if m]
                st.markdown("**Final Metrics:** " + " | ".join(metric_line))
