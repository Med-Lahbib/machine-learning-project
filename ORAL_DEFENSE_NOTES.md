# Oral Defense Presentation Notes
## Semantic Book Recommendation with Bi-directional LSTM + Triplet Loss

---

## Opening Statement (1 minute)
"Good [morning/afternoon]. My project is a **Semantic Book Recommendation System** that uses **Bi-directional LSTM** and **Triplet Loss** to find books with similar themes and descriptions. I built an end-to-end pipeline: from scraping 1,000 books, to implementing a deep metric learning model in PyTorch, and finally deploying a real-time discovery dashboard. Unlike simple keyword search, this system understands the *meaning* behind book descriptions."

---

## Section 1: Problem & Motivation (2 minutes)

### Key Points to Emphasize:
1. **Semantic Gap:** Keyword search (e.g., searching "magic") misses books that use different words for the same concept (e.g., "sorcery," "enchantment").
2. **Beyond Metadata:** Recommendations should be based on the *narrative essence* found in descriptions.
3. **Deep Metric Learning:** The goal is to map book descriptions into a high-dimensional space where "similar" books are physically close to each other.
4. **The Dataset:** 1,000 books scraped from `books.toscrape.com` with titles, prices, and full plot synopses.

---

## Section 2: Data & Feature Engineering (2 minutes)

### Key Points:
1. **Web Scraping:** Enhanced scraper using `BeautifulSoup` to visit individual book pages for full descriptions.
2. **Content-Based Ground Truth:** 
   - We use **TF-IDF (Term Frequency-Inverse Document Frequency)** as a baseline to identify "positive" pairs for training.
   - This ensures the model learns to associate books with similar themes rather than just similar star ratings.
3. **Tokenization:** Built a vocabulary of 5,000 words. Padded/Truncated descriptions to 150 tokens.

### Visual:
- Explain the **Triplet** concept:
  - **Anchor:** A book (e.g., *Sapiens*)
  - **Positive:** A similar book (e.g., *Guns, Germs, and Steel*)
  - **Negative:** A random, unrelated book (e.g., *Alice in Wonderland*)

---

## Section 3: Model Architecture (3 minutes)

### Key Points:
1. **Bi-directional LSTM (The "Encoder"):**
   - Reads text forward and backward to capture global context.
   - **Critical Implementation:** We concatenate the *final* hidden states from both directions to represent the whole sequence.
2. **Dense Projector:** 
   - A 2-layer MLP with **ReLU** activation.
   - Projects 512 LSTM units down to a 128-dimensional "Book Embedding."
3. **L2 Normalization:**
   - Ensures all embeddings sit on a hypersphere (unit length).
   - Makes similarity calculation easy (just a dot product).

### Visual:
```
Text Sequence → Embedding Layer → Bi-LSTM → Concatenation → Dense Head (ReLU) → L2 Norm → 128D Vector
```

---

## Section 4: Training with Triplet Loss (2 minutes)

### Key Points:
1. **Loss Function:** `TripletLoss` with a 0.5 margin.
2. **Learning Objective:** 
   - Pull the **Positive** embedding closer to the **Anchor**.
   - Push the **Negative** embedding at least `margin` distance away.
3. **Optimization:** Adam optimizer with early stopping based on validation loss.

### Results Table:
| Metric | Value |
|-------|----------|
| Total Parameters | 2.97 Million |
| Best Val Loss | 0.4312 |
| Dataset Size | 3,000 Triplet Pairs |

---

## Section 5: Results & Relevance Discussion (2 minutes)

### Why this model is superior to the initial version:
1. **Fixed Architecture:** Early versions had a bug extracting LSTM states; the current version uses the full bidirectional context.
2. **Semantic Understanding:** The model now groups books by *genre and theme*.
   - Example: *Sapiens* (History) → *Siddhartha* (Philosophy/History).
3. **Embedding Distribution:** We avoided "embedding collapse" where everything looks 99% similar. Now similarity scores are varied and meaningful.

---

## Section 6: Deployment (1 minute)

### Key Points:
1. **Interactive Dashboard:**
   - **Title Search:** Find books similar to ones in the database.
   - **Custom Inference:** Type your own book idea → model encodes it in real-time → finds matches.
2. **Model Transparency:**
   - Integrated a **Model Architecture Visualizer** in the app.
   - Shows layers, parameter distribution, and activation functions.

---

## Section 7: Lessons Learned (2 minutes)

1. **Architecture Accuracy:** Small indexing bugs (like LSTM state extraction) can silently cripple a model's potential.
2. **Data-Centric AI:** The biggest improvement came from changing *how* we defined "similar" books (moving from ratings to content similarity).
3. **The Hypersphere:** Understanding vector geometry (L2 normalization) is key to stable similarity search.

---

## Closing Statement (1 minute)

"In summary, I've moved beyond simple classification to create a **deep learning retrieval system**. By combining the sequential power of Bi-LSTMs with the geometric constraints of Triplet Loss, I've built a tool that can navigate the semantic space of literature. It represents a complete project lifecycle: from raw web data to a mathematical embedding space, to a user-facing AI application. Thank you."
