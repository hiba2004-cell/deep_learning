# Fake News Detection using Deep Learning

This project demonstrates **fake news detection** using multiple natural language processing (NLP) techniques, ranging from **baseline models** to **state-of-the-art transformer models**. The goal is to classify news articles as *real* or *fake*, showcasing practical deep learning skills for portfolio and CV purposes.

---

## 🚀 Models Implemented

### 1. Baseline Model (SpaCy + RandomForestClassifier, MultinomialNB, GradientBoostingClassifier)
- Converts news text into vector embeddings using **SpaCy**
- Trains a **RandomForestClassifier, MultinomialNB, GradientBoostingClassifier** classifiers, and GradientBoostingClassifier got the best performance
- Provides a **baseline performance** for comparison

### 2. Deep Learning Model (FastText + LSTM)
- Uses **pretrained FastText embeddings** to represent words
- Feeds sequences into a **BiLSTM network** for classification
- Demonstrates **deep learning on textual data**

### 3. State-of-the-Art Model (BERT Fine-Tuning)
- Fine-tunes **BERT transformer (`bert-base-uncased`)** for binary classification
- Leverages **attention mechanism** for contextual understanding
- Achieves **higher accuracy and robustness** compared to baseline and LSTM models

---

## 📊 Evaluation Metrics

- **Accuracy**: Fraction of correct predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualizes true vs predicted labels
- Optional: **ROC-AUC** for threshold performance

---
