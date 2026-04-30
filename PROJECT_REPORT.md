# Semantic Book Recommendation with Bi-directional LSTM
## Deep Learning Project Report

**Author:** Mohamed  
**Date:** April 30, 2026  
**Project Type:** Deep Learning / NLP / Metric Learning  
**Framework:** PyTorch + Streamlit

---

## 1. Executive Summary

This project implements a **Deep Metric Learning system** for book recommendations using a **Bi-directional Long Short-Term Memory (Bi-LSTM)** architecture trained with **Triplet Loss**. The model transforms raw book descriptions into 128-dimensional semantic embeddings, allowing for similarity-based retrieval. The system was trained on a dataset of 1,000 books scraped from **books.toscrape.com**. Unlike traditional keyword-based search, this model captures the semantic essence of narratives, effectively grouping books with similar themes (e.g., matching history with philosophy). The project includes a full pipeline: custom web scraping, content-based triplet generation, PyTorch model training with early stopping, and a production-ready Streamlit dashboard featuring real-time inference and architecture visualization.

---

## 2. Project Motivation and Objectives

### 2.1 Motivation
Standard recommendation systems often rely on simple metadata (genre tags) or keyword matching. Deep Metric Learning offers a more robust alternative by learning a continuous mathematical space where the distance between items represents their semantic similarity. This project explores:
- The effectiveness of **LSTMs** in encoding long-form book descriptions.
- **Triplet Loss** as a mechanism for learning relative similarity.
- Bridging the gap between raw web data and a functional AI application.

### 2.2 Objectives
1. **Enhanced Data Sourcing:** Scrape 1,000 detailed book plot synopses using multi-page pagination logic.
2. **Metric Learning Pipeline:** Implement a triplet generation strategy using TF-IDF as a semantic baseline.
3. **Architecture Optimization:** Design a Bi-LSTM encoder that correctly utilizes bidirectional context.
4. **Deployment:** Create a high-fidelity dashboard that visualizes model internals and provides real-time "custom text" search.

---

## 3. Dataset and Data Preparation

### 3.1 Data Source
- **Website:** books.toscrape.com
- **Collection:** Custom Python scraper visiting individual detail pages to capture full descriptions.
- **Volume:** 1,000 unique book entries.

### 3.2 Triplet Generation Strategy
To train a metric learning model, "triplets" are required. We generated **3,000 triplets** using the following logic:
- **Anchor:** A randomly selected book description.
- **Positive:** A book with high **TF-IDF similarity** to the anchor (ensuring shared themes).
- **Negative:** A book with low TF-IDF similarity (ensuring distinct themes).

### 3.3 Text Preprocessing
1. **Lowercase conversion** and HTML tag removal.
2. **Vocabulary:** Top 5,000 most frequent words.
3. **Padding/Truncation:** Standardized to 150 tokens per description.

---

## 4. Model Architecture

The model is a **Bi-directional LSTM Encoder** with a metric learning head:

```
[Input Text: 150 tokens]
       ↓
[Embedding Layer: 100-dim]
       ↓
[Bi-LSTM: 2 layers, 256 hidden units]
       ↓
[Concatenated Final States: 512-dim]
       ↓
[Dense Layer: 256 units + ReLU]
       ↓
[Dropout: 0.3]
       ↓
[Output Layer: 128 units]
       ↓
[L2 Normalization] → Final Embedding Vector
```

### 4.1 Key Design Choices
- **Bidirectional LSTM:** Essential for capturing dependencies in narrative text that may only become clear at the end of a sentence.
- **State Concatenation:** Correctly extracts the final hidden state from both the forward and backward passes to maximize context capture.
- **L2 Normalization:** Constrains embeddings to a unit hypersphere, making **Cosine Similarity** equivalent to the dot product, which is computationally efficient.

---

## 5. Training and Evaluation

### 5.1 Training Setup
- **Loss Function:** `TripletLoss` (Margin = 0.5).
- **Optimizer:** Adam (Learning Rate = 0.0005).
- **Regularization:** Gradient clipping (1.0) and Early Stopping.
- **Split:** 2,400 training triplets / 600 validation triplets.

### 5.2 Results
| Metric | Result |
|--------|--------|
| Total Trainable Parameters | 2,974,368 |
| Best Validation Loss | 0.4312 |
| Epochs to Convergence | 10 |

### 5.3 Performance Analysis
- **Semantic Grouping:** Qualitative analysis shows the model effectively groups books by theme. For example, history books consistently return other non-fiction or historical fiction as top results.
- **Robustness:** The model handles "Custom Text" inference gracefully, mapping fictional descriptions to relevant existing books in the database.

---

## 6. Deployment and Interactivity

### 6.1 Streamlit Dashboard Features
1. **Semantic Discovery:** Users can select a book or type a custom description to find the top 8 matches.
2. **Architecture Visualizer:** A technical breakdown showing layer flow, parameter counts, and activation functions.
3. **Parameter Distribution:** An interactive pie chart visualizing how model complexity is distributed (90% LSTM, 7% Embedding, 3% Dense Head).
4. **Metrics Viewer:** Plots real-time training history and convergence curves.

---

## 7. Technical Implementation Details

- **Deep Learning:** PyTorch
- **Web Scraping:** BeautifulSoup4, Requests
- **Data Science:** Pandas, Scikit-learn (TF-IDF, Cosine Similarity)
- **Visualization:** Plotly, Streamlit

---

## 8. Discussion and Conclusion

This project demonstrates the power of **Deep Metric Learning** for information retrieval. By moving away from a simple classification task and focusing on the geometric relationship between text embeddings, we created a system that truly "understands" narrative similarity. 

**Key Lessons:**
1. **Metric Learning > Classification:** For recommendation tasks, learning a distance metric is more scalable and flexible than predicting fixed categories.
2. **Bidirectional Context:** In NLP, the bidirectional state extraction logic is a common failure point; correctly implementing it significantly improved recommendation relevance.
3. **End-to-End Delivery:** A machine learning model is only as good as the interface that serves it. The Streamlit dashboard bridges the gap between a complex 3M-parameter model and a usable consumer tool.

---

**Report Date:** April 30, 2026  
**Model Version:** embedding_model.pt (Fixed Bi-LSTM)  
**Status:** Production Ready
