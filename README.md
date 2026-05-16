# Part 1 — Neural Network Fundamentals: Customer Churn Prediction

## Problem Statement
Binary classification task to predict whether a telecom customer
will churn based on usage patterns, billing behaviour, and
satisfaction scores.

## Dataset
- **Source:** Provided synthetic dataset
- **File:** customer_churn_nn.csv
- **Rows:** 2000 customers | **Features:** 16 input features
- **Target:** churn (1 = churned, 0 = retained)
- **Dataset Link:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs

## Approach
- Encoded categorical features using LabelEncoder
- Scaled numerical features using StandardScaler
- Built a feed-forward neural network using TensorFlow/Keras
- Ran 6 hyperparameter experiments varying depth, width,
  learning rate, batch size, and activation function

## Model Architecture
- Input → Dense(64, ReLU) → Dropout(0.2) → Dense(64, ReLU)
  → Dropout(0.2) → Dense(1, Sigmoid)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001)

## Results
See model_comparison_table.csv in results/ for full experiment
comparison. Baseline model achieved strong test accuracy with
stable convergence over 50 epochs.