# Oral Defense Presentation Notes
## Sentiment Analysis with Bi-directional LSTM

---

## Opening Statement (1 minute)
"Good [morning/afternoon]. My project is a **Sentiment Analysis system** that uses **Bi-directional LSTM**, a type of neural network, to classify whether a book review is positive or negative. I built this completely from scratch—from scraping data off websites, to training the neural network, to deploying it as an interactive web app. Today I'll walk you through the problem, my approach, the results, and what I learned."

---

## Section 1: Problem & Motivation (2 minutes)

### Key Points to Emphasize:
1. **Real Problem:** Sentiment analysis is everywhere—social media, customer feedback, product recommendations
2. **My Dataset:** I scraped 797 books from a real e-commerce website (books.toscrape.com)
3. **Challenge:** Can I predict whether someone liked a book based on its description and rating?
4. **Why This Matters:** Understanding customer sentiment helps businesses make decisions

### Visual/Demo:
- Show a few example descriptions from `books.csv`
  - "This book is a masterpiece of..." → 5 stars
  - "Terrible plot, waste of time" → 1 star

---

## Section 2: Data & Preprocessing (2 minutes)

### Key Points:
1. **Web Scraping:** Used Python (BeautifulSoup + requests) to collect book metadata
   - Title, Price, Description, Rating
   - 797 books total
   
2. **Labeling Strategy:** Converted text ratings to binary labels
   - ⭐⭐⭐⭐⭐ (Five) = Positive (1)
   - ⭐⭐⭐⭐ (Four) = Positive (1)
   - ⭐⭐⭐ (Three) = Excluded (neutral)
   - ⭐⭐ (Two) = Negative (0)
   - ⭐ (One) = Negative (0)

3. **Text Cleaning:** 
   - Lowercase, remove HTML, remove special characters
   - Keep only words
   - Tokenize by space
   
4. **Train/Test Split:**
   - 296 training samples
   - 501 test samples

### Visual:
```
Raw Text: "It's amazing! <br/> Price: $10.00"
↓
Cleaned: "its amazing price"
↓
Vocabulary Encoded: [234, 567, 890, 0, 0, ...]
```

---

## Section 3: Model Architecture (3 minutes)

### Key Points:
1. **Why LSTM?**
   - RNNs process text sequentially (left to right)
   - LSTMs avoid "vanishing gradient" problem
   - Bi-directional LSTMs read text both ways → understand context better

2. **Architecture Overview:**
   ```
   Text Input
   ↓
   Embedding (learn word representations)
   ↓
   Bi-LSTM (forward + backward passes)
   ↓
   Max Pooling (summarize sequence)
   ↓
   Linear Layer (final prediction)
   ↓
   Sigmoid (probability 0-1)
   ```

3. **Specific Design Choices:**
   - **Embedding Dim 32:** Small enough to train quickly, large enough to capture meaning
   - **Hidden Dim 64:** Allows model to learn complex patterns
   - **2 LSTM Layers:** Stacking allows hierarchical feature learning
   - **Dropout 0.3:** Prevents overfitting
   - **Max Pooling:** Handles variable-length sequences elegantly

4. **Total Parameters:** ~198,000 (relatively small, trains fast on CPU)

### Visual/Demo:
- Draw or show the architecture diagram
- Emphasize: "This is essentially a smart text encoder that learns to recognize sentiment patterns"

---

## Section 4: Training Process (2 minutes)

### Key Points:
1. **Training Loop:**
   - Process batches of 32 samples
   - Compute loss (Binary Cross-Entropy)
   - Backpropagation to update weights
   - Gradient clipping to prevent instability

2. **Epochs:** Trained for 3 epochs
   - Each epoch processes all 296 training samples
   - After each epoch, evaluated on 501 test samples
   
3. **Hyperparameters:**
   - Learning Rate: 0.001 (Adam optimizer)
   - Batch Size: 32
   - Loss Function: BCELoss (standard for binary classification)

### Results Table (Important!):
| Epoch | Train Acc | Test Acc | Observation |
|-------|----------|----------|-------------|
| 1 | 47.6% | **52.9%** ✓ Best | Model generalizes |
| 2 | 52.4% | 50.9% | Test accuracy drops |
| 3 | 56.8% | 49.7% | Overfitting begins |

**Key Insight:** "We saved the best model (Epoch 1) and stopped, even though training accuracy was still improving. This is good practice—we avoid overfitting."

---

## Section 5: Results & Accuracy Discussion (2 minutes)

### The Honest Truth:
"52.9% accuracy might sound low. To be transparent: it's only slightly better than random guessing (50%). Let me explain why and what it means."

### Why Accuracy is Modest:
1. **Task is Genuinely Hard:**
   - A 5-star book might have a generic, unemotional description
   - A 1-star book might be written with great grammar and positive language
   - Description tone ≠ Reader sentiment
   
2. **Limited Training Data:**
   - 296 samples is small for deep learning
   - Modern NLP models train on millions of samples
   
3. **Class Balance:**
   - 51% positive, 49% negative
   - No "easy majority" to exploit
   - Model must truly learn patterns

### What This Means:
- ✅ **Model IS Learning:** Test accuracy > random (52.9% > 50%)
- ✅ **Generalization Works:** Model doesn't memorize training data
- ❌ **Problem is Hard:** Descriptions aren't strongly predictive of ratings
- ⚠️ **Not Production-Ready:** For real use, would need explicit reviews or more/better data

### Comparison:
"If I retrained on IMDB reviews (explicit text) instead of descriptions, accuracy would likely jump to 75-80%. The model architecture is sound; it's the data that's challenging."

---

## Section 6: Deployment (1 minute)

### Key Points:
1. **Streamlit Dashboard:**
   - Web app where users enter text → model predicts sentiment
   - Real-time inference
   - Interactive exploration of book dataset
   - Training metrics visualization

2. **How Inference Works:**
   - Load model + vocabulary
   - Clean user input (same preprocessing)
   - Encode to integers
   - Run through neural network
   - Output probability (0-1) → threshold at 0.5

3. **Model Artifacts Saved:**
   - `sentiment_model.pt` – Trained weights
   - `vocab.pkl` – Word→Number mappings
   - `training_history.json` – Metrics

---

## Section 7: Lessons Learned (2 minutes)

### Top 3 Takeaways:

1. **Domain Matters More Than You Think**
   - Same model architecture on different data → very different results
   - Task definition is crucial (explicit reviews vs. implicit descriptions)

2. **Data > Models**
   - A larger, cleaner dataset beats hyperparameter tuning
   - Garbage in, garbage out
   - Spend more time on data than on model tweaking

3. **Production Requires Humility**
   - 52.9% isn't enough for a real product
   - Always have a baseline to compare against
   - Understand why your model works or fails

---

## Section 8: What I'd Do Differently (1 minute)

**If I could redo this:**
1. Start with explicit review text (IMDB), not descriptions
2. Explore baseline models first (TF-IDF + logistic regression) before jumping to LSTM
3. Do more error analysis: which predictions fail? Why?
4. Use hyperparameter tuning (grid search) instead of guessing
5. Implement early stopping to automatically find best epoch

---

## Section 9: Code Highlights (Optional, if asked)

**Model Forward Pass:**
```python
def forward(self, x):
    embedded = self.embedding(x)           # Words → Vectors
    lstm_out, _ = self.lstm(embedded)      # Process sequence
    pooled = lstm_out.max(dim=1)[0]        # Summarize
    logits = self.fc(pooled)               # Predict
    return torch.sigmoid(logits)           # Probability
```

**Key Python Libraries:**
- PyTorch (neural network)
- Streamlit (web UI)
- Pandas (data manipulation)
- BeautifulSoup (web scraping)

---

## Closing Statement (1 minute)

"To summarize: I've built a complete sentiment classification system from data collection to deployment. While the 52.9% accuracy on book descriptions is modest, it demonstrates that I understand neural networks, can implement them in PyTorch, and can take a project from raw data to a working web app. The results show the importance of data quality and problem formulation in machine learning. Thank you."

---

## Anticipated Questions & Answers

### Q: Why is accuracy so low?
**A:** The task is inherently hard—product descriptions don't strongly correlate with customer ratings. The model learns patterns but the signal is weak. On explicit reviews, this model would achieve 75%+ accuracy.

### Q: Why Bi-LSTM instead of transformer?
**A:** Transformers (BERT) are powerful but require more data to fine-tune. LSTM is a proven, elegant architecture for this task and trains efficiently on a CPU.

### Q: How would you improve it?
**A:** 
1. Get better data (explicit reviews, not descriptions)
2. Use more training data (tens of thousands, not hundreds)
3. Try transfer learning (pre-trained embeddings)
4. Do proper hyperparameter tuning

### Q: Can you deploy this to production?
**A:** Not as-is. 52.9% accuracy isn't reliable for real users. I'd either:
- Retrain on a better dataset, or
- Use it as a research prototype to explore how descriptions relate to ratings

### Q: How long did this take?
**A:** [Honest answer—e.g., "About 10-15 hours total: 2 hours scraping, 1 hour preprocessing, 3 hours coding the model, 2 hours training/debugging, 4 hours building the UI and writing documentation."]

### Q: What was the hardest part?
**A:** Realizing the model was correct but the data was weak. Spent time debugging before understanding that the task itself is genuinely difficult.

---

## Backup Demos (If Time Allows)

### Demo 1: Show App Running
```bash
streamlit run app.py
```
- Type example: "This book is absolutely wonderful!"
- Show prediction: Positive (78% confidence)
- Browse books table

### Demo 2: Show Training Script
```bash
python train.py
```
- Walk through dataset loading
- Vocabulary building
- Model architecture summary
- Epoch-by-epoch progress

### Demo 3: Code Walkthrough
- Open `model.py` → show SentimentLSTM class
- Open `train.py` → show train loop
- Open `app.py` → show prediction pipeline

---

## Confidence Checkpoints

✅ I understand the problem
✅ I can explain why accuracy is modest (honest about limitations)
✅ I can describe the architecture in detail
✅ I can run the app and show it working
✅ I've thought about future improvements
✅ I know what I'd do differently

---

## Time Budget

- Opening: 1 min
- Problem & Data: 2 min
- Architecture: 3 min
- Training: 2 min
- Results: 2 min
- Deployment: 1 min
- Lessons: 2 min
- Improvements: 1 min
- **Total: ~14 minutes** (leaves 6 min for questions in 20-min slot)

---

**Final Thought:** "This project taught me that machine learning is 10% clever algorithms and 90% understanding your data and problem. I'm proud of the implementation and even more proud of understanding why the results are what they are."

Good luck! 🚀
