# Emotion Classification from Tweets using SVM

This project implements a complete pipeline for classifying emotions expressed in English-language tweets using Support Vector Machines (SVMs). It includes preprocessing, model training with various SVM kernels, performance evaluation, clustering for insight, and an alert system for negative emotions.

## 📂 Dataset

The dataset (`emotions.csv`) must contain:

- `text`: The tweet content
- `label`: The corresponding emotion label (numeric or string)

## 📦 Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
✅ Preprocessing
Converts to lowercase

Removes URLs, mentions, special characters, digits

Removes English stopwords

Applies stemming

✅ SVM Classification
Evaluates kernels: linear, poly, rbf, sigmoid

Prints classification reports

Displays confusion matrices and accuracy

Visualizes performance comparison

✅ Clustering
Uses KMeans to cluster TF-IDF feature vectors

Plots distribution of clustered tweets

✅ Model Saving
Saves best SVM model, TF-IDF vectorizer, and label encoder using joblib

✅ Alerting System
Classifies new input text

Prints emotion

Raises alert for sadness, fear, or anger
