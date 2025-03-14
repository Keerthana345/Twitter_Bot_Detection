# Twitter Bot Detection

## Overview
Twitter Bot Detection is a machine learning project designed to classify Twitter accounts as either bots or humans. The project leverages both **supervised and unsupervised learning techniques** to improve detection accuracy by analyzing tweet content, user activity, and engagement metrics.

## Features
- **Real-time Twitter Data Collection**: Fetches tweets and user data using the Twitter API (Tweepy).
- **Feature Engineering**: Extracts and processes features like account age, tweet frequency, sentiment scores, and engagement ratios.
- **Supervised Learning**: Implements multiple classification models for bot detection.
- **Unsupervised Learning**: Uses clustering algorithms to detect bot behavior anomalies.
- **Model Evaluation**: Assesses performance using metrics like accuracy.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Tweepy, NLTK, Matplotlib, Seaborn, TextBlob
- **Machine Learning Models**: Logistic Regression, Random Forest, SVM, Multi-Layer Perceptron (MLP), XGBoost, DBSCAN
- **Dataset**: Publicly available Kaggle dataset + Twitter API collected data

## Dataset Details
- **Source**: Kaggle & Twitter API (Tweepy)
- **Categories**:
  - **Bots (1)**: Automated accounts generating tweets using scripts or AI.
  - **Non-Bots (0)**: Human users who manually post tweets.
- **Dataset Size**: 1020 Twitter accounts (510 Bots, 510 Non-Bots)

## Methodology
### **1. Supervised Learning Approach**
We applied various machine learning classifiers to distinguish bots from humans using labeled datasets:
- **Logistic Regression**: Predicts the probability of an account being a bot.
- **Random Forest**: An ensemble learning method improving classification accuracy.
- **Support Vector Machine (SVM)**: Identifies the optimal hyperplane for classification.
- **Multi-Layer Perceptron (MLP - Neural Network)**: Captures complex patterns in tweet data.
- **XGBoost**: A boosting algorithm optimizing classification performance.

### **Feature Engineering for Supervised Learning**
To enhance classification, we extracted key features from the dataset:
- **Account-based features**: Account age, number of followers, following ratio.
- **Tweet-based features**: Tweet length, word count, hashtags count, mentions count, URLs count, special characters count.
- **Lexical diversity**: Ratio of unique words to total words.
- **Sentiment analysis**: Uses TextBlob to analyze tweet sentiment (-1 to +1 scale).
- **Engagement metrics**: Likes, retweets, replies per tweet.

### **2. Unsupervised Learning Approach**
To identify potential bot clusters among unlabeled accounts, we used:
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
  - Clustered accounts based on extracted features.
  - Identified anomalies in engagement and tweet frequency.
  - Helped reveal hidden bot networks and flagged spam bots.

## Results
- **Supervised Learning**: Achieved high accuracy with XGBoost and Random Forest classifiers.
- **Unsupervised Learning**: Successfully identified hidden bot clusters using DBSCAN.
- **Feature Engineering**: Significantly improved model performance by incorporating user engagement and tweet-based metrics.

## Future Enhancements
- Integrate deep learning models for improved accuracy.
- Expand dataset with real-time Twitter data.
- Deploy the model as a web app for live bot detection.
