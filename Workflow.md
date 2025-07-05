🗓️ 7-Day Plan
Day 1: Data Sourcing & Setup
Dataset: Use the Cornell BirdCLEF dataset or Xeno-Canto.

Metadata: Includes latitude, altitude, season, time, etc.

Tasks:

Download a subset of the dataset.

Convert bird call audio files into spectrograms (Mel-spectrograms using librosa or torchaudio).

Preprocess metadata using scikit-learn (StandardScaler, OneHotEncoder, etc.).

Day 2: LightGBM Modeling on Tabular Metadata
Train a baseline LightGBM classifier using tabular features.

Perform cross-validation.

Use Optuna to tune hyperparameters (e.g., num_leaves, learning_rate, max_depth).

Evaluate using accuracy, F1-score, ROC-AUC.

Day 3–4: CNN Modeling on Spectrograms
Train a CNN or use a pretrained model (e.g., ResNet, EfficientNet) on the spectrograms.

Framework: TensorFlow or PyTorch.

Apply data augmentation (e.g., time masking, frequency masking).

Save intermediate embeddings if you plan to fuse models.

Day 5: Fusion Model (Advanced but optional)
Extract features from your CNN (penultimate layer).

Concatenate with tabular features.

Train a shallow classifier (LightGBM or MLP) on fused features.

Evaluate improvement over individual models.

Day 6: Model Evaluation & Interpretability
Metrics: Confusion matrix, accuracy, per-species F1 scores.

Visualization:

Grad-CAM for CNN-based spectrograms.

SHAP plots for LightGBM model.

Identify most important metadata features and frequency patterns.

Day 7: Reporting & GitHub
Organize your code into modules (e.g., preprocessing.py, train_tabular.py, train_audio.py).

Create a clean README.md:

Problem

Datasets

Models

Evaluation metrics

Insights

Push to GitHub for portfolio showcase.